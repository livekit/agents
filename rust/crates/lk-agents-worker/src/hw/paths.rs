//! Resolve the process's own cgroups through procfs, including nonstandard mounts and namespaces.

use std::path::{Path, PathBuf};

use super::cgroup::{CpuMonitorKind, cpu_count_v2};

pub(super) struct CpuCgroup {
    pub kind: CpuMonitorKind,
    pub usage: PathBuf,
    cpu: Mount,
    cpuset: Option<Mount>,
}

#[derive(Clone)]
struct Mount {
    directory: PathBuf,
    root: PathBuf,
}

impl Mount {
    fn ancestors(&self) -> impl Iterator<Item = &Path> {
        self.directory.ancestors().take_while(|p| p.starts_with(&self.root))
    }
}

impl CpuCgroup {
    pub fn discover() -> Option<Self> {
        Self::resolve(
            &std::fs::read_to_string("/proc/self/cgroup").ok()?,
            &std::fs::read_to_string("/proc/self/mountinfo").ok()?,
        )
    }

    fn resolve(membership: &str, mounts: &str) -> Option<Self> {
        // In hybrid setups CPU can still belong to v1 despite a unified mount being present.
        if let (Some(cpu), Some(usage)) = (
            resolve_mount(membership, mounts, Some("cpu")),
            resolve_mount(membership, mounts, Some("cpuacct")),
        ) {
            return Some(Self {
                kind: CpuMonitorKind::CgroupV1,
                usage: usage.directory.join("cpuacct.usage"),
                cpu,
                cpuset: resolve_mount(membership, mounts, Some("cpuset")),
            });
        }
        if let Some(cpu) = resolve_mount(membership, mounts, None) {
            return Some(Self {
                kind: CpuMonitorKind::CgroupV2,
                usage: cpu.directory.join("cpu.stat"),
                cpuset: Some(cpu.clone()),
                cpu,
            });
        }
        None
    }

    pub fn cpu_count(&self, system: f32) -> f32 {
        self.capacity(system, |p| std::fs::read_to_string(p).ok())
    }

    fn capacity(&self, system: f32, read: impl Fn(&Path) -> Option<String>) -> f32 {
        let mut count = system;
        // A child's unlimited quota still inherits every visible ancestor's limit.
        for directory in self.cpu.ancestors() {
            let limit = if self.kind == CpuMonitorKind::CgroupV2 {
                read(&directory.join("cpu.max")).map(|s| cpu_count_v2(None, &s, system))
            } else {
                let quota = read(&directory.join("cpu.cfs_quota_us"))
                    .and_then(|s| s.trim().parse::<f32>().ok());
                let period = read(&directory.join("cpu.cfs_period_us"))
                    .and_then(|s| s.trim().parse::<f32>().ok());
                match (quota, period) {
                    (Some(q), Some(p)) if q > 0.0 && p > 0.0 => Some(q / p),
                    _ => None,
                }
            };
            if let Some(limit) = limit.filter(|n| n.is_finite() && *n > 0.0) {
                count = count.min(limit);
            }
        }
        if let Some(cpuset) = &self.cpuset {
            for directory in cpuset.ancestors() {
                for name in ["cpuset.cpus.effective", "cpuset.effective_cpus", "cpuset.cpus"] {
                    if let Some(n) = read(&directory.join(name)).and_then(|s| cpuset_count(&s)) {
                        count = count.min(n as f32);
                    }
                }
            }
        }
        count
    }
}

fn cpuset_count(raw: &str) -> Option<u32> {
    let mut count = 0_u32;
    for range in raw.trim().split(',') {
        let (start, end) = range.split_once('-').unwrap_or((range, range));
        let start = start.parse::<u32>().ok()?;
        let end = end.parse::<u32>().ok()?;
        count = count.checked_add(end.checked_sub(start)?.checked_add(1)?)?;
    }
    (count > 0).then_some(count)
}

fn unescape(raw: &str) -> PathBuf {
    PathBuf::from(
        raw.replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\"),
    )
}

fn resolve_mount(membership: &str, mounts: &str, controller: Option<&str>) -> Option<Mount> {
    let path = membership.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':');
        fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        let matches =
            controller.map_or(controllers.is_empty(), |c| controllers.split(',').any(|v| v == c));
        matches.then(|| PathBuf::from(path))
    })?;
    // Never let a procfs namespace escape its cgroup mount.
    if path.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return None;
    }
    mounts
        .lines()
        .filter_map(|line| {
            let (left, right) = line.split_once(" - ")?;
            let fields: Vec<_> = left.split_whitespace().collect();
            let mut fs = right.split_whitespace();
            let kind = fs.next()?;
            fs.next()?;
            let options = fs.next()?;
            match controller {
                None if kind != "cgroup2" => return None,
                Some(c) if kind != "cgroup" || !options.split(',').any(|v| v == c) => return None,
                _ => {}
            }
            let mounted_root = unescape(fields.get(3)?);
            let root = unescape(fields.get(4)?);
            let relative = path
                .strip_prefix(&mounted_root)
                .ok()
                // A private cgroup namespace reports `/` while mountinfo may retain the host root.
                .or_else(|| (path == Path::new("/")).then(|| Path::new("")))?;
            Some((
                mounted_root.components().count(),
                Mount { directory: root.join(relative), root },
            ))
        })
        .max_by_key(|(specificity, _)| *specificity)
        .map(|(_, mount)| mount)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_v2_uses_membership_ancestor_quota_and_effective_cpuset() {
        let group =
            CpuCgroup::resolve("0::/team/worker\n", "1 0 0:1 / /custom rw - cgroup2 cgroup rw\n")
                .unwrap();
        assert_eq!(group.usage, Path::new("/custom/team/worker/cpu.stat"));
        let read = |p: &Path| match p.to_str().unwrap() {
            "/custom/team/worker/cpu.max" => Some("max 100000".into()),
            "/custom/team/cpu.max" => Some("50000 100000".into()),
            "/custom/team/worker/cpuset.cpus.effective" => Some("2-3,6".into()),
            _ => None,
        };
        assert_eq!(group.capacity(16.0, read), 0.5);
        assert_eq!(
            group.capacity(16.0, |p| {
                (p.file_name().unwrap() == "cpuset.cpus.effective").then(|| "2-3,6".into())
            }),
            3.0
        );
    }

    #[test]
    fn v1_separate_mounts_preserve_fractional_quota_and_unlimited_capacity() {
        let group = CpuCgroup::resolve(
            "0::/\n2:cpu:/jobs/w\n3:cpuacct:/jobs/w\n",
            "1 0 0:1 /jobs /quota rw - cgroup cgroup rw,cpu\n2 0 0:2 /jobs /usage rw - cgroup cgroup rw,cpuacct\n3 0 0:3 / /unified rw - cgroup2 cgroup rw\n",
        ).unwrap();
        assert_eq!(group.usage, Path::new("/usage/w/cpuacct.usage"));
        assert_eq!(
            group.capacity(8.0, |p| match p.file_name()?.to_str()? {
                "cpu.cfs_quota_us" => Some("50000".into()),
                "cpu.cfs_period_us" => Some("100000".into()),
                _ => None,
            }),
            0.5
        );
        assert_eq!(group.capacity(8.0, |_| None), 8.0);
    }

    #[test]
    fn fractional_multi_cpu_limits_are_not_rounded_down() {
        let group = CpuCgroup::resolve("0::/\n", "1 0 0:1 / /cg rw - cgroup2 cgroup rw\n").unwrap();
        for (quota, expected) in
            [("50000 100000", 0.5), ("150000 100000", 1.5), ("450000 100000", 4.5)]
        {
            assert_eq!(
                group.capacity(8.0, |p| {
                    (p.file_name().unwrap() == "cpu.max").then(|| quota.to_owned())
                }),
                expected
            );
        }
    }

    #[test]
    fn private_namespace_and_escaped_mountpoint_resolve_safely() {
        let mounts = "1 0 0:1 /host/container /my\\040cgroup rw - cgroup2 cgroup rw\n";
        let group = CpuCgroup::resolve("0::/\n", mounts).unwrap();
        assert_eq!(group.usage, Path::new("/my cgroup/cpu.stat"));
        assert!(CpuCgroup::resolve("0::/../elsewhere\n", mounts).is_none());
    }
}
