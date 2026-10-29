//! The prompts the workflows' host tools are written with, in the `workflow_prompts` module.

#[crate::host(file(module = "workflow_prompts"))]
const _: () = ();

/// `EndCallTool`'s description, before its `extra_description`.
#[crate::host(constant)]
pub const END_CALL_DESCRIPTION: &str = "
Ends the current call and disconnects immediately.

Call when:
- The user clearly indicates they are done (e.g., “that’s all, bye”).

Do not call when:
- The user asks to pause, hold, or transfer.
- Intent is unclear.

This is the final action the agent can take.
Once called, no further interaction is possible with the user.
Don't generate any other text or response when the tool is called.
";
