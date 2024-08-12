import asyncio
import multiprocessing as mp
import os
import socket


async def async_send(loop, sock, message):
    await loop.sock_sendall(sock, message.encode("utf-8"))


async def async_recv(loop, sock, buffer_size=1024):
    data = await loop.sock_recv(sock, buffer_size)
    return data.decode("utf-8")


def worker_process(send_sock):
    # This will run in the worker process
    loop = asyncio.get_event_loop()

    async def worker_task():
        # Simulate sending messages from the worker to the main process
        for i in range(5):
            message = f"Message {i} from process {os.getpid()}"
            print(f"Sending: {message}")
            await async_send(loop, send_sock, message)
            await asyncio.sleep(1)

    loop.run_until_complete(worker_task())
    send_sock.close()


async def main():
    parent_sock, child_sock = socket.socketpair()

    ctx = mp.get_context("spawn")
    process = ctx.Process(target=worker_process, args=(child_sock,))
    process.start()

    child_sock.close()  # Close the child socket in the main process

    loop = asyncio.get_event_loop()

    # Asynchronously receive messages from the worker process
    async def receive_messages():
        while True:
            message = await async_recv(loop, parent_sock)
            if not message:
                break
            print(f"Received: {message}")

    await receive_messages()

    # Wait for the process to finish
    process.join()
    parent_sock.close()


if __name__ == "__main__":
    asyncio.run(main())
