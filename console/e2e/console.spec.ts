import { mkdirSync, writeFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";

/**
 * Against the live stack of examples/ (livekit-server, agent-db, the
 * registry, the fast brain and the slow brain, `agentdb debug`): real models,
 * so nothing here compares the agent's wording. The checks read states, the
 * metadata, and the views of the session's database.
 * Screenshots go to E2E_SHOTS (default /tmp/lk-console) for both engines,
 * named by the viewport's size.
 */

const SHOTS = process.env.E2E_SHOTS ?? "/tmp/lk-console";
mkdirSync(SHOTS, { recursive: true });

/** A turn of the conversation can take the slow brain a while on a real model. */
const TURN = 120_000;

async function shot(page: Page, name: string) {
  const size = page.viewportSize()!;
  await page.screenshot({ path: `${SHOTS}/${test.info().project.name}-${size.width}x${size.height}-${name}.png` });
}

/** Nothing may overflow the stage sideways. */
async function noOverflow(page: Page) {
  const widths = await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]);
  expect(widths[0], "horizontal overflow").toBe(widths[1]);
}

async function open(page: Page) {
  await page.goto("/");
  const input = page.getByLabel("Message");
  await expect(input).toBeEnabled({ timeout: 60_000 });
  return input;
}

/** Types a message and sends it with Enter: the composer's only control. */
async function say(page: Page, text: string) {
  const before = await page.locator("[data-entry='status'][data-state]").count();
  await page.getByLabel("Message").fill(text);
  await page.getByLabel("Message").press("Enter");
  /* The turn is over when its task's end state is written under it. */
  await expect(page.locator("[data-entry='status'][data-state]")).toHaveCount(before + 1, { timeout: TURN });
  return page.locator("[data-entry='status'][data-state]").nth(before);
}

/** The conversation as shown, one line per entry, for the report. */
async function transcript(page: Page) {
  const entries = page.locator("[data-entry]");
  const count = await entries.count();
  const lines: string[] = [];
  for (let index = 0; index < count; index++) {
    const entry = entries.nth(index);
    const kind = await entry.getAttribute("data-entry");
    const voice = (await entry.getAttribute("data-voice")) ? " (voice)" : "";
    const text = (await entry.innerText()).replace(/\s+/g, " ").trim();
    lines.push(`${kind}${voice}: ${text}`);
  }
  return lines;
}

function tab(page: Page, view: string) {
  return page.locator(`[data-panel-tab='${view}']`);
}

/** Counts what would mean a call was made without the speech button: token requests and signalling sockets. */
function watchConnections(page: Page) {
  const seen = { tokens: 0, sockets: 0 };
  page.on("request", (request) => {
    /* The console's own route, not the debug API's grant (GET /api/token?sub=). */
    const url = new URL(request.url());
    if (request.method() === "POST" && url.pathname === "/api/token" && url.port === "3000") seen.tokens += 1;
  });
  page.on("websocket", (socket) => {
    if (socket.url().includes("/rtc")) seen.sockets += 1;
  });
  return seen;
}

test("a stale session is dropped quietly; the debug drawer opens on a tab and closes on a second click; nothing connects before the speech button", async ({ page }) => {
  /* What a fresh profile with a session from another stack sees: the saved
     id answers 404, the drawer was left on Stream, which needs a selection. */
  await page.addInitScript(() => {
    window.localStorage.setItem("console.session", "AT_stale0000000000");
    window.localStorage.setItem("console.drawer.view", "stream");
  });
  const seen = watchConnections(page);
  await open(page);
  await expect(page.locator("[data-invitation]")).toBeVisible();
  await expect(page.locator("[data-entry]")).toHaveCount(0);
  await expect(page.locator("[data-drawer]")).toHaveAttribute("data-drawer", "closed");
  await expect(page.locator("[data-panel='sessions']")).toBeVisible();
  /* The orb is the mark, before any call. */
  await expect(page.locator("[data-head] [data-orb]")).toBeVisible();
  /* The stale id is forgotten, and nothing on the page says so. */
  await expect.poll(() => page.evaluate(() => window.localStorage.getItem("console.session"))).toBeNull();
  await expect(page.locator("body")).not.toContainText(/404|not_found|twirp|error/i);
  /* The composer sits with the invitation in the middle of the stage, not at the foot. */
  const composer = (await page.locator("[data-composer]").boundingBox())!;
  expect(composer.y).toBeLessThan(900 * 0.65);
  expect(composer.width).toBeLessThanOrEqual(720);
  await noOverflow(page);
  await shot(page, "empty");

  /* A tab opens the drawer; a second click on it closes it. */
  await tab(page, "lanes").click();
  await expect(page.locator("[data-drawer]")).toHaveAttribute("data-drawer", "lanes");
  await tab(page, "lanes").click();
  await expect(page.locator("[data-drawer]")).toHaveAttribute("data-drawer", "closed");
  await expect(page.locator("[data-pane='drawer']")).toHaveCount(0);
  /* Selecting a conversation does not connect anything either. */
  const rows = page.locator("[data-session]");
  if ((await rows.count()) > 1) {
    await rows.nth(1).click();
    await expect(rows.nth(1)).toHaveAttribute("data-selected", "true");
    await page.locator("[data-action='new-conversation']").click();
    await expect(page.locator("[data-invitation]")).toBeVisible();
  }
  await page.waitForTimeout(2_000);
  expect(seen.tokens, "token requests before the speech button").toBe(0);
  expect(seen.sockets, "room sockets before the speech button").toBe(0);

  await page.getByLabel("Message").fill("Hello");
  await page.getByLabel("Message").press("Enter");
  await expect(page.locator("[data-entry='user']")).toHaveCount(1);
  /* The head says Thinking until the first token; a fast model may already be answering. */
  await expect(page.locator("[data-state-line='thinking'], [data-entry='agent']").first()).toBeVisible();
  await shot(page, "thinking");
  await expect(page.locator("[data-invitation]")).toHaveCount(0);
  /* The same composer, now at the foot; the list names the conversation by its first message for now. */
  const moved = (await page.locator("[data-composer]").boundingBox())!;
  expect(moved.y).toBeGreaterThan(900 * 0.8);
  await expect(page.locator("[data-session][data-selected='true'] [data-session-title]")).toHaveText("Hello");
  await expect(page.locator("[data-entry='status'][data-state]")).toHaveCount(1, { timeout: TURN });
  /* Idle in text, the head says nothing under the name. */
  await expect(page.locator("[data-state-line]")).toHaveCount(0);
  /* The console named the session after its first turn: the title reaches the list and the tab. */
  await expect(page.locator("[data-session][data-selected='true'][data-titled='true']")).toHaveCount(1, { timeout: 20_000 });
  const titled = (await page.locator("[data-session][data-selected='true'] [data-session-title]").innerText()).trim();
  expect(titled).not.toBe("New conversation");
  await expect(page).toHaveTitle(titled);
  /* The session id in Lanes copies on one click and says so. */
  const sessionId = await page.evaluate(() => window.localStorage.getItem("console.session"));
  await tab(page, "lanes").click();
  const copy = page.locator(`[data-session-summary] [data-copy='${sessionId}']`);
  await copy.click();
  await expect(copy).toHaveAttribute("data-copied", "true");
  await expect(copy).not.toHaveAttribute("data-copied", "true", { timeout: 5_000 });
  expect(seen.tokens, "token requests without a call").toBe(0);
  await noOverflow(page);
});

test("three turns: the slow brain's progress, a booking that asks which doctor, the answer; the views of the session", async ({ page }) => {
  await open(page);

  await page.getByLabel("Message").fill("Check my insurance: Aetna member 12345");
  await page.getByLabel("Message").press("Enter");
  /* The slow brain's portal steps arrive as working statuses before the answer: the card opens on the first, the state line carries the latest. */
  await expect(page.locator("[data-entry='delegation'] [data-progress]").first()).toBeVisible({ timeout: TURN });
  await expect(page.locator("[data-entry='delegation']")).toHaveAttribute("data-delegation-state", /submitted|working/);
  await shot(page, "progress");
  const first = page.locator("[data-entry='status'][data-state]").first();
  await expect(first).toHaveAttribute("data-state", "completed", { timeout: TURN });
  await expect(page.locator("[data-entry='agent']").first()).not.toBeEmpty();
  /* The delegation card: the slow brain over A2A, at least one progress line, completed, its task id the slow brain's. */
  const card = page.locator("[data-entry='delegation']").first();
  await expect(card).toBeVisible();
  await expect(card).toHaveAttribute("data-delegation-state", "completed", { timeout: 15_000 });
  await expect(card).toHaveAttribute("data-delegation-task", /^AT|^[A-Za-z0-9_-]{6,}/, { timeout: 15_000 });
  const expertTask = (await card.getAttribute("data-delegation-task"))!;
  await expect(card.locator("[data-badge]")).toHaveText("A2A");
  /* Done, the card is one line; unfolded, the progress it spoke and the transport in its footer. */
  if ((await card.getAttribute("data-open")) === "false") await card.locator("[data-action='unfold']").click();
  expect(await card.locator("[data-progress]").count()).toBeGreaterThan(0);

  const second = await say(page, "I'd like to book a visit");
  /* The slow brain's tool waits on the person (plan 7.4): its task ends input-required, the card
     says so, and the front desk relays the question in its own completed turn. */
  await expect(second).toHaveAttribute("data-state", "completed");
  await expect(page.locator("[data-entry='delegation']").nth(1)).toHaveAttribute("data-delegation-state", "input-required");
  await expect(page.locator("[data-entry='agent']").last()).toContainText(/doctor|Lin|Patel|Okafor/i);

  const third = await say(page, "Dr. Lin");
  await expect(third).toHaveAttribute("data-state", "completed");
  expect(await page.locator("[data-entry='agent']").count()).toBeGreaterThanOrEqual(2);
  await noOverflow(page);
  await shot(page, "conversation");

  /* Tasks: three on the fast brain's lane, and one on the slow brain's per turn the front desk took to it. */
  await tab(page, "tasks").click();
  await expect(page.locator("[data-task][data-lane='fast-brain']")).toHaveCount(3, { timeout: 10_000 });
  await expect(page.locator("[data-task][data-lane='slow-brain']")).toHaveCount(3);
  await expect(page.locator(`[data-task='${expertTask}'][data-lane='slow-brain']`)).toHaveCount(1);
  await expect(page.locator("[data-task][data-status='failed'], [data-task][data-status='canceled']")).toHaveCount(0);
  await shot(page, "tasks");

  /* Lanes: the slow brain's lane holds the patient it learned; its state folds open under the row. */
  await tab(page, "lanes").click();
  await expect(page.locator("[data-lane='slow-brain']")).toBeVisible({ timeout: 10_000 });
  await page.locator("[data-lane='slow-brain']").click();
  await expect(page.locator("[data-panel='lanes'] pre").first()).toContainText("insurance");

  /* Journal: three closed runs, each a bracket; both rows of a run carry the same mark. */
  await tab(page, "journal").click();
  await expect(page.locator("[data-event='run_started']").nth(2)).toBeVisible({ timeout: 10_000 });
  await expect(page.locator("[data-event='run_started'] [data-run-mark='ended']").nth(2)).toBeVisible({ timeout: 10_000 });
  await expect(page.locator("[data-event='run_ended'] [data-run-mark='ended']").nth(2)).toBeVisible();
  await expect(page.locator("[data-run-mark='open']")).toHaveCount(0);
  await expect(page.locator("[data-event='run_started'] [data-rail='start']").first()).toBeVisible();
  /* The table: a row opens its payload under it, one at a time; the filter narrows the rows; a header sorts. */
  const journal = page.locator("[data-table='journal']");
  await journal.locator("[data-event='run_started']").first().click();
  await expect(journal.locator("[data-fold-of]")).toHaveCount(1);
  await journal.locator("[data-event='run_ended']").first().click();
  await expect(journal.locator("[data-fold-of]")).toHaveCount(1);
  await expect(journal.locator("[data-fold-of]")).toContainText("outcome");
  const endedRows = await journal.locator("[data-event='run_ended']").count();
  await journal.locator("[data-table-filter]").fill("run_ended");
  await expect(journal.locator("[role='row'][data-row]")).toHaveCount(endedRows);
  await journal.locator("[data-table-filter]").fill("");
  await journal.locator("[data-column='id']").click();
  await expect(journal.locator("[data-column='id']")).toHaveAttribute("aria-sort", "ascending");
  await journal.locator("[data-column='id']").click();
  await expect(journal.locator("[data-column='id']")).toHaveAttribute("aria-sort", "descending");
  await expect(journal.locator("[role='row'][data-row]").first()).toHaveAttribute("data-row", String(await journal.locator("[role='row'][data-row]").count()));
  await journal.locator("[data-column='id']").click();
  /* A header boundary drags the column's width. */
  const grip = journal.locator("[data-grip='type']");
  const gripBox = (await grip.boundingBox())!;
  const typeBefore = (await journal.locator("[data-column='type']").boundingBox())!.width;
  await page.mouse.move(gripBox.x + gripBox.width / 2, gripBox.y + gripBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(gripBox.x + gripBox.width / 2 + 40, gripBox.y + gripBox.height / 2, { steps: 4 });
  await page.mouse.up();
  expect((await journal.locator("[data-column='type']").boundingBox())!.width).toBeGreaterThan(typeBefore + 30);
  await expect(journal.locator("[data-column='type']")).toHaveAttribute("aria-sort", "none");
  await shot(page, "journal");

  /* The task id under a turn copies on click; the glyph after it opens Tasks on that row. */
  const taskId = (await first.locator("[data-task-link]").getAttribute("data-task-link"))!;
  await first.locator(`[data-copy='${taskId}']`).click();
  await expect(first.locator(`[data-copy='${taskId}']`)).toHaveAttribute("data-copied", "true");
  await first.locator("[data-task-link]").click();
  await expect(page.locator("[data-panel='tasks']")).toBeVisible();
  await expect(page.locator(`[data-task='${taskId}'][data-selected='true']`)).toHaveCount(1);
  /* From there, the Stream view replays it from the start. */
  await page.locator("[data-action='open-stream']").click();
  await expect(page.locator("[data-panel='stream']")).toBeVisible();
  await expect(page.locator("[data-panel='stream']")).toContainText(taskId.slice(-5));
  await expect(page.locator("[data-stream-state='closed']")).toBeVisible({ timeout: 15_000 });
  expect(Number(await page.locator("[data-stream-frames]").getAttribute("data-stream-frames"))).toBeGreaterThan(0);
  await shot(page, "stream");
  await tab(page, "info").click();
  await expect(page.locator("[data-panel='info']")).toBeVisible();
  await shot(page, "info");
  await tab(page, "info").click();

  const lines = await transcript(page);
  writeFileSync(`${SHOTS}/${test.info().project.name}-conversation.txt`, lines.join("\n"));
  console.log(lines.join("\n"));
});

test("a reload mid-reply resumes the stream through tasks/resubscribe", async ({ page }) => {
  await open(page);
  await page.getByLabel("Message").fill("Check my insurance: Aetna member 12345");
  await page.getByLabel("Message").press("Enter");
  /* The first progress line means the task exists and its reply is in flight. */
  await expect(page.locator("[data-entry='delegation'] [data-progress]").first()).toBeVisible({ timeout: TURN });
  await page.reload();
  await expect(page.locator("[data-entry='status']").filter({ hasText: "resuming" })).toBeVisible({ timeout: 60_000 });
  await expect(page.locator("[data-entry='status'][data-state='completed']")).toBeVisible({ timeout: TURN });
  await expect(page.locator("[data-entry='agent']").last()).not.toBeEmpty();
  /* And the session came back from the registry, selected. */
  await expect(page.locator("[data-session][data-selected='true']")).toHaveCount(1);
  await shot(page, "resumed");
  console.log((await transcript(page)).join("\n"));
});

test("the lanes view shows the lease a turn takes and frees", async ({ page }) => {
  await open(page);
  await page.locator("[data-action='new-conversation']").click();
  await tab(page, "lanes").click();
  await expect(page.locator("[data-panel='lanes']")).toBeVisible();
  await say(page, "Hello");
  /* The lane's lease holder is `<worker id>/<handle>`. */
  await expect(page.locator("[data-lane='fast-brain'] [data-lease-holder]")).toHaveAttribute("data-lease-holder", /^AW_\S+\//, { timeout: 10_000 });
  await expect(page.locator("[data-lane='fast-brain'] [data-lease-state]")).toHaveAttribute("data-lease-state", "free");
  await expect(page.locator("[data-session-summary] [data-session-id]")).toHaveText(/^AT_/);
  /* The front desk keeps no state: its row has nothing to fold open. */
  await expect(page.locator("[data-fold-toggle='fast-brain']")).toBeDisabled();
  await shot(page, "lanes");
});

/**
 * The drawer's edge is a hairline with a wide grab area. The drawer must
 * follow the pointer while it is held, not jump on release, and its height
 * must come back after a reload; its tab closes it and Cmd+K starts over.
 */
test("the debug drawer resizes live, its height survives a reload, Cmd+K starts a new conversation", async ({ page }) => {
  await open(page);
  await tab(page, "lanes").click();
  const pane = page.locator("[data-pane='drawer']");
  const edge = page.locator("[data-splitter='drawer']");
  const box = (await edge.boundingBox())!;
  const height = async () => Math.round((await pane.boundingBox())!.height);
  const start = await height();

  await page.mouse.move(box.x + 200, box.y + box.height / 2);
  await page.mouse.down();
  for (let step = 1; step <= 4; step++) {
    await page.mouse.move(box.x + 200, box.y + box.height / 2 - step * 30);
    await page.waitForTimeout(50);
  }
  const dragged = await height();
  await page.mouse.up();
  expect(dragged).toBeGreaterThan(start + 100);
  expect(await height()).toBe(dragged);

  await page.reload();
  await expect(page.getByLabel("Message")).toBeEnabled({ timeout: 60_000 });
  await expect(page.locator("[data-panel='lanes']")).toBeVisible();
  expect(Math.abs((await height()) - dragged)).toBeLessThan(4);
  await tab(page, "lanes").click();
  await expect(pane).toHaveCount(0);
  /* Cmd+. brings it back on the last view. */
  await page.keyboard.press("ControlOrMeta+.");
  await expect(page.locator("[data-panel='lanes']")).toBeVisible();
  /* Cmd+K starts a new conversation. */
  await page.keyboard.press("ControlOrMeta+k");
  await expect(page.locator("[data-invitation]")).toBeVisible();
});

/**
 * Voice. On chromium the fake microphone plays the caller's line (a visit
 * with Dr. Lin on Friday); the agent must acknowledge, speak the slow brain's
 * progress and come back with a concrete slot, and the slow brain's lane must
 * appear in Lanes. On webkit, which has no fake device, the call connects
 * without a microphone and the states are checked without audio.
 */
test("a call from the speech button: the same conversation, spoken", async ({ page, browserName }) => {
  const seen = watchConnections(page);
  await open(page);
  await page.locator("[data-action='new-conversation']").click();
  await expect(page.locator("[data-invitation]")).toBeVisible();
  await page.waitForTimeout(1_500);
  expect(seen.tokens, "token requests before the speech button").toBe(0);
  expect(seen.sockets, "room sockets before the speech button").toBe(0);

  await page.locator("[data-action='speech']").click();
  await expect(page.locator("[data-composer]")).toHaveAttribute("data-calling", /connecting|connected/);
  await expect(page.locator("[data-composer]")).toHaveAttribute("data-calling", "connected", { timeout: 60_000 });
  expect(seen.tokens).toBe(1);
  await expect(page.locator("[data-call-bar]")).toBeVisible();
  await expect(page.locator("[data-panel='conversation']")).toHaveAttribute("data-mode", "voice");
  /* The orb hears the call. */
  await expect(page.locator("[data-head] [data-orb]")).toBeVisible();
  await expect(page.locator("[data-call-elapsed]")).not.toHaveText("0:00", { timeout: 10_000 });
  /* The agent greets: its words arrive as a spoken turn. */
  await expect(page.locator("[data-entry='agent'][data-voice='true']").first()).toBeVisible({ timeout: 60_000 });
  /* Text stays available during the call. */
  await expect(page.getByLabel("Message")).toBeEnabled();

  if (browserName === "chromium") {
    /* The caller's line, as the realtime model heard it. */
    await expect(page.locator("[data-entry='user'][data-voice='true']").first()).toBeVisible({ timeout: 60_000 });
    await shot(page, "voice");
    /* After it: the acknowledgement, at least one progress line, the answer with a slot. */
    const after = () => page.locator("[data-entry='user'][data-voice='true'] ~ [data-entry='agent'][data-voice='true']");
    /* The realtime model may fold the acknowledgement and the progress into one line, so the
       count is a floor; the delegation card holds the progress either way. */
    await expect
      .poll(async () => after().count(), { timeout: 150_000 })
      .toBeGreaterThanOrEqual(2);
    /* On a call the card is fed from the slow brain's task row, whose progress lives on its stream, so
       the card's state is the assertion here; the text path checks the progress lines. */
    await expect(page.locator("[data-entry='delegation'] [data-state='completed']").first()).toBeVisible({ timeout: 150_000 });
    await expect
      .poll(async () => (await after().allInnerTexts()).join(" | "), { timeout: 150_000 })
      .toMatch(/\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?|AM|PM|o'clock)|(Lin|Patel|Okafor)/i);
    /* The slow brain wrote its own lane of the call's session. */
    await tab(page, "lanes").click();
    await expect(page.locator("[data-lane='slow-brain']")).toBeVisible({ timeout: 30_000 });
    await expect(page.locator("[data-lane='fast-brain']")).toBeVisible();
    await shot(page, "voice-lanes");
    await page.setViewportSize({ width: 2000, height: 1500 });
    await shot(page, "voice");
    await noOverflow(page);
    await page.setViewportSize({ width: 1440, height: 900 });
  } else {
    await shot(page, "voice");
  }

  const voiceLines = await transcript(page);
  writeFileSync(`${SHOTS}/${browserName}-voice.txt`, voiceLines.join("\n"));
  console.log(voiceLines.join("\n"));

  /* Hanging up returns to text; the turns remain. */
  const spoken = await page.locator("[data-entry][data-voice='true']").count();
  await page.locator("[data-action='hang-up']").click();
  await expect(page.locator("[data-composer]")).toHaveAttribute("data-calling", "idle", { timeout: 15_000 });
  await expect(page.locator("[data-call-bar]")).toHaveCount(0);
  await expect(page.locator("[data-head] [data-orb]")).toBeVisible();
  expect(await page.locator("[data-entry][data-voice='true']").count()).toBe(spoken);
  /* A reload during or after a call never rejoins: the call is over. */
  const before = seen.tokens;
  await page.reload();
  await expect(page.getByLabel("Message")).toBeEnabled({ timeout: 60_000 });
  await page.waitForTimeout(2_000);
  expect(seen.tokens).toBe(before);
  await expect(page.locator("[data-composer]")).toHaveAttribute("data-calling", "idle");
});

test("at 956px the conversation keeps inside the stage", async ({ page }) => {
  await page.setViewportSize({ width: 956, height: 900 });
  await open(page);
  const composer = (await page.locator("[data-composer]").boundingBox())!;
  expect(composer.x).toBeGreaterThanOrEqual(0);
  expect(composer.x + composer.width).toBeLessThanOrEqual(956);
  await noOverflow(page);
  await shot(page, "narrow");
});

/** The same states at the larger size, for the eye. */
test("the views at 2000x1500", async ({ page }) => {
  await page.setViewportSize({ width: 2000, height: 1500 });
  await open(page);
  await page.locator("[data-action='new-conversation']").click();
  await expect(page.locator("[data-invitation]")).toBeVisible();
  await shot(page, "empty");
  await page.getByLabel("Message").fill("Check my insurance: Aetna member 12345");
  await page.getByLabel("Message").press("Enter");
  await expect(page.locator("[data-state-line='thinking']")).toBeVisible();
  await shot(page, "thinking");
  await expect(page.locator("[data-entry='delegation'] [data-progress]").first()).toBeVisible({ timeout: TURN });
  await shot(page, "progress");
  await expect(page.locator("[data-entry='status'][data-state]")).toHaveCount(1, { timeout: TURN });
  await expect(page.locator("[data-entry='delegation']")).toHaveAttribute("data-delegation-state", "completed", { timeout: 15_000 });
  await noOverflow(page);
  await shot(page, "conversation");
  for (const view of ["lanes", "tasks", "journal", "inbox", "info"]) {
    await tab(page, view).click();
    await expect(page.locator(`[data-panel='${view}']`)).toBeVisible();
    await shot(page, view);
  }
  await noOverflow(page);
});
