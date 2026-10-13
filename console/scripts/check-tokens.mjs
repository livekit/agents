#!/usr/bin/env node
/*
 * Build guard. tokens.css is the only file allowed to contain a raw color,
 * a font stack, or a raw transition duration. Everything else must go
 * through a var(--token).
 *
 * This guard has caught real regressions in every previous iteration of
 * this UI. Do not weaken it; add a token instead.
 */
import { readFileSync } from "node:fs";
import { readdir } from "node:fs/promises";
import path from "node:path";

const ROOT = path.resolve(import.meta.dirname, "..");
const TOKENS_FILE = path.join(ROOT, "src/app/tokens.css");
const SKIP_DIRS = new Set([
  "node_modules",
  ".next",
  ".git",
  "dist",
  "build",
  "test-results",
  "playwright-report",
]);
const SCAN_EXT = new Set([".css", ".ts", ".tsx", ".mjs", ".js"]);
const PROSE_EXT = new Set([".md"]);

const RULES = [
  {
    name: "raw hex color",
    re: /(?<![&\w])#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b/g,
  },
  { name: "raw rgb()/rgba()", re: /\brgba?\(/g },
  { name: "raw hsl()/hsla()", re: /\bhsla?\(/g },
  { name: "raw oklch()/oklab()", re: /\bokl(?:ch|ab)\(/g },
  {
    name: "named CSS color",
    re: /(?<![\w-])(?:white|black|red|green|blue|gray|grey|silver|crimson|amber|orange|yellow|purple|teal|navy)(?=\s*[;,)])/gi,
  },
  {
    name: "font stack outside tokens",
    re: /font-family\s*:(?![^;]*var\(--font-)/g,
  },
  {
    name: "raw transition/animation duration",
    re: /(?:transition|animation)(?:-duration)?\s*:[^;]*\b\d+m?s\b/g,
  },
  /*
   * Tailwind moves the same mistake into class names. `bg-[#303030]` is a raw
   * colour with a different syntax, and `text-[11px]` re-types a value the
   * theme already names. An arbitrary value that reads a token, such as
   * `grid-cols-[calc(var(--h-header)+var(--pad-2))]`, is fine, which is why
   * the rule only fires when there is no var() inside the brackets.
   */
  {
    name: "arbitrary colour in a Tailwind class",
    re: /-\[(?:#[0-9a-fA-F]{3,8}|(?:rgb|hsl|okl(?:ch|ab))a?\([^\]]*\))\]/g,
  },
  {
    name: "hard-coded length in a Tailwind class",
    re: /(?<![\w-])(?:[whpm]|max-[whw]|min-[whw]|gap|text|rounded|top|left|right|bottom|inset|size)(?:[xytrbles])?-\[(?![^\]]*var\(--)[^\]]*\d(?:px|rem|em)\]/g,
  },
];

// No em dashes anywhere in this app, prose included: the rule for this
// repository. Checked on every scanned file, the markdown files too.
const EM_DASH = { name: "em dash", re: /\u2014/g };

// Lines carrying this marker are exempt, with a reason. Use sparingly.
const ALLOW = /guard-ok:/;

async function walk(dir, out = []) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    if (e.name.startsWith(".") && e.name !== ".storybook") continue;
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (SKIP_DIRS.has(e.name)) continue;
      await walk(full, out);
    } else if (SCAN_EXT.has(path.extname(e.name)) || PROSE_EXT.has(path.extname(e.name))) {
      out.push(full);
    }
  }
  return out;
}

const files = (await walk(ROOT)).filter(
  (f) => f !== path.join(ROOT, "scripts/check-tokens.mjs"),
);

const violations = [];
for (const file of files) {
  const prose = PROSE_EXT.has(path.extname(file));
  /* tokens.css is where the raw values live; it only answers for em dashes. */
  const rules = prose || file === TOKENS_FILE ? [EM_DASH] : [...RULES, EM_DASH];
  const lines = readFileSync(file, "utf8").split("\n");
  lines.forEach((line, i) => {
    if (ALLOW.test(line)) return;
    for (const rule of rules) {
      rule.re.lastIndex = 0;
      const m = rule.re.exec(line);
      if (m) {
        violations.push({
          file: path.relative(ROOT, file),
          line: i + 1,
          rule: rule.name,
          text: line.trim().slice(0, 100),
        });
        break;
      }
    }
  });
}

if (violations.length) {
  console.error(
    `\ntokens guard: ${violations.length} violation(s). Add a token in src/app/tokens.css and use var(--token).\n`,
  );
  for (const v of violations) {
    console.error(`  ${v.file}:${v.line}  [${v.rule}]`);
    console.error(`    ${v.text}`);
  }
  console.error("");
  process.exit(1);
}

console.log(`tokens guard: clean (${files.length} files scanned)`);
