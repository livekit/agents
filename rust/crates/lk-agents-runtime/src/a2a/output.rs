use super::{ExactOutput, HandlerTurn, Report};
use lk_agents_abi::errors::HostError;
use lk_agents_abi::{BoxFuture, ChatMessage};
use lk_agents_net::signal;

impl HandlerTurn {
    pub(super) fn enqueue_exact(
        &self,
        message: ChatMessage,
    ) -> BoxFuture<'static, Result<(), HostError>> {
        let (delivered, receipt) = signal::once();
        let output = Report::Exact(ExactOutput {
            task_id: self.task_id.clone(),
            message: Box::new(message),
            delivered,
        });
        let (admitted, after) = signal::once();
        let before = self.output_tail.lock().replace(after);
        let reports = self.reports.clone();
        let work = async move {
            if let Some(before) = before {
                match before.wait().await {
                    Ok(()) => {}
                    Err(signal::Ended) => return,
                }
            }
            reports.send(output).await;
        };
        match self.outputs.upgrade() {
            Some(owner) => {
                if !owner.spawn_settling(
                    &*self.spawner,
                    "A2A exact output admission",
                    admitted,
                    work,
                ) {
                    return Box::pin(async {
                        Err(HostError::new("the A2A output owner has closed"))
                    });
                }
            }
            None => {
                drop(admitted);
                drop(work);
            }
        }
        Box::pin(async move {
            receipt
                .wait()
                .await
                .map_err(|_| HostError::new("the A2A output owner ended before exact delivery"))?
        })
    }

    /// Admission failure belongs to its private reader; final business output still proceeds.
    /// Waiting here preserves report order, including when no caller awaited its exact output.
    pub(super) async fn wait_output_admissions(&self) {
        let tail = self.output_tail.lock().clone();
        if let Some(tail) = tail {
            match tail.wait().await {
                Ok(()) | Err(signal::Ended) => {}
            }
        }
    }
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
