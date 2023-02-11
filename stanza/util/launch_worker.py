import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from stanza.logging import logger
from stanza.util.pool import WorkerRemote
import argparse
import asyncio

import rpyc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=os.environ.get("REPLICA", None), required=False)
    parser.add_argument("--host", default=os.environ.get("RPC_HOST", None))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RPC_PORT", "18861")))
    parser.add_argument("--verbose", default=bool(os.environ.get("VERBOSE", False)), required=False)
    args = parser.parse_args()

    if args.verbose:
        logger.trace("worker", f"Worker {args.id} connecting to {args.host}:{args.port}")

    conn = rpyc.connect(args.host, args.port)
    worker = WorkerRemote(args.id, asyncio.new_event_loop())
    conn.root.register_worker(worker)
    try:
        conn.serve_all()
    except KeyboardInterrupt:
        logger.info("worker", f"Worker {args.id} shutting down...")

if __name__=="__main__":
    main()