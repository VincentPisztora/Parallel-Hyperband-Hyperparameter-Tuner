# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 13:36:44 2026

@author: pisztov1
"""

import pathlib
import time
import os
import random

def acquire_lock(file_path, timeout=30, retry_delay=0.5):
    """
    Acquires an advisory lock using a lock file.
    """
    lock_path = pathlib.Path(str(file_path) + ".lock")
    start_time = time.time()

    while True:
        if not lock_path.exists():
            try:
                # Attempt to create the lock file exclusively
                # Use 'x' mode which fails if the file already exists
                with open(lock_path, 'x') as f:
                    f.write(f"Locked by PID: {os.getpid()}")
                print(f"Lock acquired: {lock_path}")
                return lock_path
            except FileExistsError:
                # Another process created the file in the meantime
                pass
        
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Could not acquire lock for {file_path} within {timeout} seconds.")
        
        print(f"Waiting for lock on {file_path}...")
        retry_delay_r = random.uniform(.01,retry_delay)
        time.sleep(retry_delay_r)

def release_lock(lock_path):
    """
    Releases the lock by removing the lock file.
    """
    try:
        lock_path.unlink()
        print(f"Lock released: {lock_path}")
    except FileNotFoundError:
        print(f"Lock file not found (maybe another process removed it?): {lock_path}")

def lock_wrapper(file_to_access_path,work_function):
    file_to_access = pathlib.Path(file_to_access_path)
    try:
        lock = acquire_lock(file_to_access)
        
        # Simulate file access and work
        print(f"Accessing {file_to_access}...")
        
        print(f'Starting work')
        out = work_function()
        print(f'Work completed')
        
        return out
        
    except TimeoutError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Ensure the lock is released even if an error occurs
        if 'lock' in locals() and lock.exists():
            release_lock(lock)



















