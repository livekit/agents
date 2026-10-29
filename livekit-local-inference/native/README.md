# Native inference host

We implement the public Node-API operations used by LiveKit's 0.2.7 inference
addon. The addon owns the model and each VAD stream's recurrent state. The host
retains its object and calls each method with its original receiver.

## Ownership

One Python 3.15.2 child owns one loaded addon image and one thread-confined
environment. VAD tokens identify at most 128 live contexts. Dropping a token
releases its object; closing the environment retires any remaining contexts.

## Parent loss

Start the parent watcher with a dedicated read-only pipe before loading the
addon. Its native thread exits only its own child on EOF or descriptor failure,
including while a foreign callback holds the GIL. Keep the writer in the parent
until the child is reaped; stop the watcher only after closing the environment.

## Proof seam

The source-owned fake exercises the same callbacks, promises, references and
finalizers as the addon. Its recurrent-state fixture proves ownership and reset
isolation; model quality is checked separately against captured genuine outputs.
The frozen public headers and their complete license notices live beside this
source.
