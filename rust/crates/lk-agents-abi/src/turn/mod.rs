//! Turn taking (`voice/turn.py`): the end-of-turn detector seam, its wire and prediction, the
//! adaptive-interruption seam, and keyterm detection.

pub mod detector;
pub mod eot;
pub mod events;
pub mod interruption;
pub mod keyterms;
