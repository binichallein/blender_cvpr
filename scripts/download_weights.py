"""Resume a ModelScope download after transient failures; verified shards stay cached."""
import argparse
import os
import socket
import ipaddress
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--range-parallel', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--attempts', type=int, default=30)
    parser.add_argument('--resolve', action='append', default=[], metavar='HOST=IP',
                        help='Optional process-local DNS override; TLS still verifies the original host')
    args = parser.parse_args()
    # Adopt a download already in progress without concurrently writing its files.
    while args.wait_pid:
        process = Path('/proc') / str(args.wait_pid)
        try:
            command = process.joinpath('cmdline').read_bytes().decode().split('\0')
            active = process.joinpath('stat').read_text().split()[2] != 'Z'
        except FileNotFoundError:
            break
        if not active or args.model not in command:
            break
        time.sleep(10)
    addresses = {}
    for override in args.resolve:
        host, address = override.split('=', 1)
        ipaddress.ip_address(address)
        addresses[host] = address
    if addresses:
        original_resolver = socket.getaddrinfo
        def resolver(host, *positional, **keywords):
            return original_resolver(addresses.get(host, host), *positional, **keywords)
        socket.getaddrinfo = resolver
        print('PROCESS_LOCAL_DNS', addresses, flush=True)
    os.environ['MODELSCOPE_DOWNLOAD_PARALLELS'] = str(args.range_parallel)
    import modelscope.hub.file_download as download
    from modelscope import snapshot_download
    download.API_FILE_DOWNLOAD_TIMEOUT = args.timeout
    for attempt in range(1, args.attempts + 1):
        try:
            print('DOWNLOAD_ATTEMPT', attempt, args.model, flush=True)
            path = snapshot_download(args.model, cache_dir=args.cache_dir, max_workers=args.workers)
            print('MODEL_READY', path, flush=True)
            return
        except Exception as error:
            print('DOWNLOAD_RETRY', type(error).__name__, str(error), flush=True)
            if attempt == args.attempts:
                raise
            time.sleep(5)


if __name__ == '__main__':
    main()
