import * as core from "@actions/core";
import { getExecOutput } from "@actions/exec";
import { getPackages } from "@manypkg/get-packages";

async function runDevSnapshotPublish() {
  const cwd = process.cwd();

  let changesetPublishOutput = await getExecOutput(
    "pnpm ci:publish",
    ["--tag dev", "--no-git-tag"],
    {
      cwd,
    }
  );

  let { packages } = await getPackages(cwd);
  let releasedPackages = [];

  let newTagRegex = /New tag:\s+(@[^/]+\/[^@]+|[^/]+)@([^\s]+)/;
  let packagesByName = new Map(packages.map((x) => [x.packageJson.name, x]));

  for (let line of changesetPublishOutput.stdout.split("\n")) {
    let match = line.match(newTagRegex);
    if (match === null) {
      continue;
    }
    let pkgName = match[1];
    let pkg = packagesByName.get(pkgName);
    if (pkg === undefined) {
      throw new Error(
        `Package "${pkgName}" not found.` +
          "This is probably a bug in the action, please open an issue"
      );
    }
    releasedPackages.push(pkg);
  }
  if (releasedPackages.length) {
    return {
      published: true,
      publishedPackages: releasedPackages.map((pkg) => ({
        name: pkg.packageJson.name,
        version: pkg.packageJson.version,
      })),
    };
  }

  return { published: false };
}

const result = await runDevSnapshotPublish();
if (result.published) {
  core.setOutput("published", "true");
  core.setOutput("publishedPackages", JSON.stringify(result.publishedPackages));
}
