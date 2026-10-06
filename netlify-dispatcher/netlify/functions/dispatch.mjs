// Momentum Alpha exact-time dispatcher (Netlify scheduled function).
//
// GitHub's `schedule:` trigger ran the 2:45 PM CT jobs 1-5 hours late from
// May-Oct 2026. This function fires on Netlify's scheduler and calls GitHub's
// workflow_dispatch API at the exact minute.
//
// Netlify cron is UTC-only, so the function is scheduled across every UTC
// hour a slot could fall in (CDT or CST) and decides what to run from the
// America/Chicago wall clock. DST is handled here, not with duplicate crons.
//
// Env vars (Netlify → Site configuration → Environment variables):
//   GH_DISPATCH_TOKEN  fine-grained PAT, momentum-system only, Actions: read/write
//   GH_REPO            optional, default connormortoncbtx-sketch/momentum-system
//   GH_REF             optional, default main
//   NTFY_TOPIC         optional, ntfy.sh topic for failure alerts (same as NTFY_CHANNEL)

const DAYS = { Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5 };

// [days, "HH:MM" Chicago time, workflow file]
// Scripts decide holiday fallbacks (Tue entry / Thu exit); we just fire both days.
export const SCHEDULE = [
  [["Mon", "Tue"], "06:00", "premarket_monitor.yml"],
  [["Mon", "Tue", "Wed", "Thu", "Fri"], "08:30", "alpaca_monitor.yml"],
  [["Mon", "Tue", "Wed", "Thu", "Fri"], "11:15", "alpaca_monitor.yml"],
  [["Mon", "Tue", "Wed", "Thu", "Fri"], "14:00", "alpaca_monitor.yml"],
  [["Mon", "Tue"], "14:45", "alpaca_entry.yml"],
  [["Mon", "Tue"], "15:10", "alpaca_place_stops.yml"],
  [["Thu", "Fri"], "14:45", "alpaca_exit.yml"],
];

// Chicago weekday + time, minute floored to the 5-min slot grid so a
// late-by-a-few-seconds invocation still matches its slot.
export function chicagoSlot(date = new Date()) {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone: "America/Chicago",
      weekday: "short",
      hour: "2-digit",
      minute: "2-digit",
      hourCycle: "h23",
    })
      .formatToParts(date)
      .map((p) => [p.type, p.value])
  );
  const minute = Math.floor(Number(parts.minute) / 5) * 5;
  return { day: parts.weekday, hhmm: `${parts.hour}:${String(minute).padStart(2, "0")}` };
}

export function dueWorkflows(date = new Date()) {
  const { day, hhmm } = chicagoSlot(date);
  return SCHEDULE.filter(([days, t]) => days.includes(day) && t === hhmm).map(([, , wf]) => wf);
}

async function alert(title, body) {
  const topic = process.env.NTFY_TOPIC;
  if (!topic) return;
  try {
    await fetch(`https://ntfy.sh/${topic}`, {
      method: "POST",
      headers: { Title: title, Priority: "high", Tags: "rotating_light" },
      body,
    });
  } catch (e) {
    console.error("ntfy alert failed:", e);
  }
}

async function dispatch(workflow) {
  const repo = process.env.GH_REPO || "connormortoncbtx-sketch/momentum-system";
  const ref = process.env.GH_REF || "main";
  const res = await fetch(
    `https://api.github.com/repos/${repo}/actions/workflows/${workflow}/dispatches`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.GH_DISPATCH_TOKEN}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "momentum-alpha-dispatcher",
      },
      body: JSON.stringify({ ref }),
    }
  );
  if (res.status !== 204) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text.slice(0, 300)}`);
  }
}

export default async () => {
  const now = new Date();
  const slot = chicagoSlot(now);
  const due = [...new Set(dueWorkflows(now))];
  console.log(`Chicago ${slot.day} ${slot.hhmm} -> ${due.length ? due.join(", ") : "nothing due"}`);

  if (due.length && !process.env.GH_DISPATCH_TOKEN) {
    await alert("Dispatcher misconfigured", "GH_DISPATCH_TOKEN is not set; nothing dispatched.");
    return new Response("missing token", { status: 500 });
  }

  const failures = [];
  for (const wf of due) {
    try {
      await dispatch(wf);
      console.log(`  dispatched ${wf}`);
    } catch (e) {
      console.error(`  FAILED ${wf}: ${e.message}`);
      failures.push(`${wf}: ${e.message}`);
    }
  }
  if (failures.length) {
    // 401/403 here usually means the token expired -- renew it.
    await alert("Dispatch FAILED", `${slot.day} ${slot.hhmm} CT\n${failures.join("\n")}`);
  }
  return new Response(JSON.stringify({ slot, due, failures }), {
    status: failures.length ? 500 : 200,
    headers: { "Content-Type": "application/json" },
  });
};

// Every UTC minute a Chicago slot can land on, across both CDT (UTC-5) and
// CST (UTC-6): 06:00 -> 11/12 UTC ... 15:10 -> 20/21 UTC. Weekdays only in
// UTC terms is safe: all slots are 6 AM-3:10 PM Chicago, same UTC date.
export const config = {
  schedule: "0,10,15,30,45 11-21 * * 1-5",
};
