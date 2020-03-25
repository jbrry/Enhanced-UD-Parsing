#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# For Python 2-3 compatible code
# https://python-future.org/compatible_idioms.html

from __future__ import print_function

import hashlib
import os
import subprocess
import sys
import time

import utilities

def print_usage(last_arg = None):
    if last_arg:
        last_arg_name, last_arg_description = last_arg
        if last_arg_name:
            last_arg_name = ' ' + last_arg_name
    else:
        last_arg_name, last_arg_description = '', ''
    print('Usage: %s [options]%s' %(os.path.split(sys.argv[0])[-1], last_arg_name))
    if last_arg_description:
        print(last_arg_description)
    print("""
Options:

    --deadline  HOURS       Do not start working on another task after
                            HOURS hours since the start of this script
                            and quit.
                            (Default: 0.0 = no limit)

    --stopfile  FILE        Do not start working on another task and
                            quit if FILE exists.
                            (Default: empty string = no stop file)

    --max-idle  HOURS       Wait no longer than HOURS hours since the
                            end of the last task (or, if there was no
                            task yet, since the start of the script)
                            for an eligible task.
                            Set to 0 for no limit on idle time.
                            (Default: 0.25 = 15 minutes)

""")


def get_training_schedule(epochs = 60):
    ''' return a udpipe-future learning rate schedule
        and epochs specification like
        "30:1e-3,5:6e-4,5:4e-4,5:3e-4,5:2e-4,10:1e-4"
        adjusted to the given number of epochs
    '''
    if type(epochs) is str:
        epochs = int(epochs)
    if 'EUD_DEBUG' in os.environ \
    and os.environ['EUD_DEBUG'].lower() not in ('0', 'false'):
        epochs = 1 + int(epochs/20)
    if 'UDPIPE_FUTURE_EPOCHS' in os.environ:
        epochs = int(os.environ['UDPIPE_FUTURE_EPOCHS'])
    if epochs < 1:
        raise ValueError('Need at least 1 epoch to train a model.')
    ref_remaining = 60
    epochs_remaining = epochs
    components = []
    for ref_count, learning_rate in [
        (30, '1e-3'),
        ( 5, '6e-4'),
        ( 5, '4e-4'),
        ( 5, '3e-4'),
        ( 5, '2e-4'),
        (10, '1e-4'),
    ]:
        n = int(epochs_remaining*ref_count/ref_remaining)
        ref_remaining -= ref_count
        epochs_remaining -= n
        if n > 0:
            components.append('%d:%s' %(n, learning_rate))
    return ','.join(components)

def memory_error(model_dir):
    if not os.path.exists('%s/stderr.txt' %model_dir):
        # problem is somewhere in the wrapper script
        return False
    f = open('%s/stderr.txt' %model_dir)
    found_oom = False
    while True:
        line = f.readline()
        if not line:
            break
        if 'ran out of memory trying to allocate' in line:
            found_oom = True
            break
    f.close()
    return found_oom

def incomplete(model_dir):
    if not os.path.exists('%s/checkpoint' %model_dir):
        return True
    return False

def my_makedirs(required_dir):
    utilities.makedirs(required_dir)

def run_command(
    command, queue_name = 'udpf', requires = None, priority = 50,
    submit_and_return = False,
    cleanup = None,
):
    task = Task(command, queue_name, requires, priority = priority)
    if submit_and_return:
        task.submit()
        task.cleanup_object = cleanup
        return task
    else:
        task.run()
        if cleanup is not None:
            cleanup.cleanup()

def wait_for_tasks(task_list):
    for task in task_list:
        if task is None:
            continue
        task.wait()
        try:
            cleanup = task.cleanup_object
        except:
            cleanup = None
        if cleanup is not None:
            cleanup.cleanup()

class Task:

    def __init__(self, command, queue_name = 'udpf', requires = None, priority = 50):
        self.command = [utilities.bstring(x) for x in command]
        self.queue_name = utilities.bstring(queue_name)
        if 'EUD_TASK_DIR' in os.environ:
            task_dir = utilities.bstring(os.environ['EUD_TASK_DIR'])
            self.queue_dir = b'/'.join((task_dir, self.queue_name))
        self.requires = []
        if requires:
            for filename in requires:
                self.requires.append(utilities.bstring(filename))
        if 'EUD_TASK_PATIENCE' in os.environ:
            patience = float(os.environ['EUD_TASK_PATIENCE'])
        else:
            patience = 48 * 3600.0
        self.expires = time.time() + patience
        self.priority = int(priority)
        if 'EUD_TASK_POLL_FREQUENCY' in os.environ:
            self.poll_frequency = float(os.environ['EUD_TASK_POLL_FREQUENCY'])
        else:
            self.poll_frequency = 1.0
        assert self.priority >= 0
        assert self.priority < 1000

    def __repr__(self):
        parts = []
        parts.append('<%s' %self.__class__.__name__)
        try:
            parts.append(utilities.std_string(self.task_id))
        except:
            pass
        parts.append('on queue %r' %self.queue_name)
        return (' '.join(parts))+'>'

    def add_required_file(self, filename):
        self.requires.append(utilities.bstring(filename))

    def required_files_exist(self, exception_if_not = False):
        for filename in self.requires:
            if not os.path.exists(filename):
                if exception_if_not:
                    raise ValueError('Missing file %r' %filename)
                return False
        return True

    def run(self):
        self.submit()
        self.wait()

    def process(self):
        self.start_processing()
        self.wait_for_processing()

    def start_processing(self):
        if self.required_files_exist(exception_if_not = True):
            subprocess.call(self.command)

    def wait_for_processing(self):
        return

    def finished_processing(self):
        return True

    def get_task_id(self):
        if 'EUD_TASK_EPOCH' in os.environ:
            t0 = float(os.environ['EUD_TASK_EPOCH'])
        else:
            t0 = 0.0
        command_fingerprint = hashlib.sha256(b'\n'.join(self.command)).hexdigest()
        hostname = 'unknown'
        for envkey in 'HOSTNAME SLURMD_NODENAME SLURM_JOB_NODELIST'.split():
            try:
                hostname = os.environ[envkey].replace('-', '_'),
                break
            except KeyError:
                pass
        task_id = utilities.bstring('%02d-%05x-%s-%s-%d-%s-%s' %(
            self.priority,
            int((time.time()-t0)/60.0),
            hostname,
            os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else 'na',
            os.getpid(),
            command_fingerprint[:8],
            command_fingerprint[8:16],
        ))
        if 'EUD_TASK_BUCKETS' in os.environ:
            num_task_buckets = int(os.environ['EUD_TASK_BUCKETS'])
        else:
            num_task_buckets = 20
        self.task_fingerprint = hashlib.sha512(task_id).hexdigest()
        self.my_task_bucket = int(self.task_fingerprint, 16) % num_task_buckets
        self.task_id = task_id
        return task_id

    def get_expiry(self):
        return self.expires

    def get_submit_name(self, task_id, my_task_bucket):
        inbox_dir = b'/'.join((self.queue_dir, b'inbox'))
        my_makedirs(inbox_dir)
        filename = b'%s/%s-%d.task' %(
            inbox_dir,
            task_id,
            my_task_bucket,
        )
        self.submit_name = filename
        return filename

    def submit(self):
        if 'EUD_TASK_DIR' not in os.environ:
            print('Running', self.command)
            sys.stderr.flush()
            sys.stdout.flush()
            self.process()
            return
        # prepare task submission file
        task_id  = self.get_task_id()
        filename = self.get_submit_name(task_id, self.my_task_bucket)
        expires  = self.get_expiry()
        f = open(filename+b'.prep', 'wb')
        f.write(b'expires\t%.1f\n' %expires)
        for required_file in self.requires:
            f.write(b'requires\t%s\n' %required_file)
        f.write(b'\n')
        f.write(b'\n'.join(self.command))
        f.write(b'\n')
        f.close()
        os.rename(filename+b'.prep', filename)
        self.submit_time = time.time()
        print('Submitted task %s to run command %r' %(task_id, self.command))
        sys.stderr.flush()
        sys.stdout.flush()

    def wait(self):
        if 'EUD_TASK_DIR' not in os.environ:
            return
        expires          = self.expires
        my_task_bucket   = self.my_task_bucket
        queue_dir        = self.queue_dir
        submit_time      = self.submit_time
        task_fingerprint = self.task_fingerprint
        task_id          = self.task_id
        # wait for task to start
        iteration = 0
        fp_length = len(task_fingerprint)
        verbosity_interval = 3600.0
        if 'EUD_DEBUG' in os.environ:
            verbosity_interval /= 100.0
        next_verbose = submit_time + verbosity_interval
        start_time_interval = (submit_time, time.time())
        while time.time() < expires and os.path.exists(self.submit_name):
            duration = 30.0 + int(task_fingerprint[iteration % fp_length], 16)
            duration = duration / self.poll_frequency
            now = time.time()
            start_time_interval = (now, now+duration)
            time.sleep(duration)
            iteration += 1
            now = time.time()
            if now >= next_verbose:
                print('Waited %.1f hours so far for task %s to start' %(
                    (now-submit_time)/3600.0,
                    task_id,
                ))
                verbosity_interval *= 1.4
                next_verbose += verbosity_interval
        # did it expire?
        has_expired = True
        # check for file in active queue
        time.sleep(5.0/self.poll_frequency) # just in case moving the file is not atomic
        filename = b'%s/active/%d/%s.task' %(
            queue_dir,
            my_task_bucket,
            task_id,
        )
        if os.path.exists(filename):
            has_expired = False
            # expectation value of start time assuming uniform
            # distribution within above sleep interval
            start_time = sum(start_time_interval) / 2.0
            # wait for task to finish (no timeout)
            verbosity_interval = 3600.0
            if 'EUD_DEBUG' in os.environ:
                verbosity_interval /= 100.0
            next_verbose = min(next_verbose, start_time + verbosity_interval)
            while os.path.exists(filename):
                duration = 30.0 + int(task_fingerprint[iteration % fp_length], 16)
                duration = duration / self.poll_frequency
                time.sleep(duration)
                iteration += 1
                now = time.time()
                if now >= next_verbose:
                    print('Task %s is running about %.1f hours so far' %(
                        task_id,
                        (now-start_time)/3600.0,
                    ))
                    verbosity_interval *= 1.4
                    next_verbose += verbosity_interval
        # task is not active --> check for completion
        filename = b'%s/completed/%d/%s.task' %(
            queue_dir,
            my_task_bucket,
            task_id,
        )
        if not os.path.exists(filename):
            print('Task %s failed' %utilities.std_string(task_id))
            raise ValueError('Task %s marked by task master as no longer active but not as complete' %utilities.std_string(task_id))
        if 'EUD_TASK_ARCHIVE_COMPLETED' in os.environ  and \
        os.environ['EUD_TASK_ARCHIVE_COMPLETED'].lower() not in ('0', 'false'):
            archive_dir = b'/'.join((queue_dir, b'archive'))
            my_makedirs(archive_dir)
            counter = 1
            while True:
                archive_name = b'%s/%s-run%03d.task' %(archive_dir, task_id, counter)
                if os.path.exists(archive_name):
                    counter += 1
                    continue
                os.rename(filename, archive_name)
                break
        elif 'EUD_TASK_CLEANUP_COMPLETED' in os.environ  and \
        os.environ['EUD_TASK_CLEANUP_COMPLETED'].lower() not in ('0', 'false'):
            os.unlink(filename)

def main(
    queue_name = 'udpf',
    task_processor = None,
    opt_deadline = None, opt_stopfile = None, opt_debug = False,
    opt_max_idle = 900.0,
    last_arg = None,
    extra_kw_parameters = {},
    callback = None,
):
    opt_help = False
    opt_debug = False
    opt_deadline = None
    opt_stopfile = None
    opt_max_idle = 900.0
    while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
        option = sys.argv[1]
        option = option.replace('_', '-')
        del sys.argv[1]
        if option in ('--help', '-h'):
            opt_help = True
            break
        elif option == '--debug':
            opt_debug = True
        elif option == '--deadline':
            opt_deadline = 3600.0 * float(sys.argv[1])
            if opt_deadline:
                opt_deadline += time.time()
            del sys.argv[1]
        elif option == '--max-idle':
            opt_max_idle = 3600.0 * float(sys.argv[1])
            del sys.argv[1]
        elif option == '--stopfile':
            opt_stopfile = sys.argv[1]
            del sys.argv[1]
        else:
            print('Unsupported or not yet implemented option %s' %option)
            opt_help = True
            break
    if len(sys.argv) != 1:
        opt_help = True
    if opt_help:
        print_usage(last_arg = last_arg)
        sys.exit(0)
    worker(
        queue_name, task_processor,
        opt_deadline, opt_stopfile, opt_debug,
        opt_max_idle,
        last_arg,
        extra_kw_parameters,
        callback,
    )

class TaskQueue:

    def __init__(self, inbox_dir, active_dir, delete_dir, task_processor, extra_kw_parameters):
        if task_processor is None:
            raise ValueError('Missing task processor')
        self.inbox_dir = inbox_dir
        self.active_dir = active_dir
        self.delete_dir = delete_dir
        self.task_processor = task_processor
        self.extra_kw_parameters = extra_kw_parameters
        self.filename2requires = {}

    def pick_task(self, opt_ignore_expiry = False, opt_debug = False):
        candidate_tasks = []
        filename2requires = {}
        for filename in os.listdir(self.inbox_dir):
            if filename.endswith(b'.task') and b'-' in filename:
                priority = filename[:2]
                candidate_tasks.append((priority, utilities.random(), filename))
                if filename in self.filename2requires:
                    required_files = self.filename2requires[filename]
                    filename2requires[filename] = required_files
        self.filename2requires = filename2requires
        candidate_tasks.sort()
        if opt_debug:
            print('Candidate tasks in inbox:', len(candidate_tasks))
        for _, _, filename in candidate_tasks:
            task_id, task_bucket = filename[:-5].rsplit(b'-', 1)
            eligible = 'unknown'
            if filename in filename2requires:
                required_files = filename2requires[filename]
                eligible = 'yes'
                for required_file in required_files:
                    if not os.path.exists(required_file):
                        eligible = 'no'
                        break
            if eligible == 'no':
                if opt_debug:
                    print('Task %s not eligible to run' %task_id)
                continue
            # read contents of task file
            taskfile = b'/'.join((self.inbox_dir, filename))
            try:
                f = open(taskfile, 'rb')
            except IOError:
                if opt_debug:
                    print('Task %s claimed by another worker or not readable' %task_id)
                continue
            if opt_debug:
                print('Reading details of task', task_id)
            delete_task = False
            expires = 0.0
            lcode = None
            required_files = []
            while True:
                line = f.readline().rstrip()
                if not line: # empty line or EOF
                    break
                fields = line.rstrip().split(b'\t')
                if len(fields) != 2:
                    print('Deleting malformed task', task_id)
                    delete_task = True
                elif line.startswith(b'expires'):
                    expires = float(fields[1])
                    if time.time() > expires and not opt_ignore_expiry:
                        print('Deleting expired task', task_id)
                        delete_task = True
                elif line.startswith(b'lcode'):
                    lcode = fields[1]
                elif line.startswith(b'requires') and eligible == 'unknown':
                    required_file = fields[1]
                    required_files.append(required_file)
                    if not os.path.exists(required_file):
                        eligible = 'no'
                if delete_task:
                    break
            if delete_task:
                f.close()
                delete_name = '%s/%s' %(self.delete_dir, filename)
                try:
                    os.rename(taskfile, delete_name)
                except:
                    print('Task %s claimed by other worker' %utilities.std_string(task_id))
                if opt_debug:
                    print('Deleted task', task_id)
                continue
            # keep list of required files to avoid reading this task file
            # again until all required files are there
            self.filename2requires[filename] = required_files
            if eligible == 'no':
                f.close()
                if opt_debug:
                    print('Task %s not eligible to run' %task_id)
                continue
            command = f.read().split(b'\n')
            f.close()
            # try to claim this task
            bucket_dir  = b'/'.join((self.active_dir, task_bucket))
            my_makedirs(bucket_dir)
            active_name = b'%s/%s.task' %(bucket_dir, task_id)
            try:
                os.rename(taskfile, active_name)
            except:
                print('Task %s claimed by other worker' %utilities.std_string(task_id))
                continue
            if opt_debug:
                print('Successfully claimed task', task_id)
            # handle last line with linebreak
            if command and command[-1] == b'':
                del command[-1]
            # found the first task eligible to run
            task = self.task_processor(command, **self.extra_kw_parameters)
            task.active_name = active_name
            task.submit_time = os.path.getmtime(active_name)
            task.task_id = task_id
            task.task_bucket = task_bucket
            task.expires = expires
            if lcode:
                task.lcode = lcode
            return task
        if opt_debug:
            print('No eligible task found')
        return None

def worker(
    queue_name = 'udpf',
    task_processor = None,
    opt_deadline = None, opt_stopfile = None, opt_debug = False,
    opt_max_idle = 900.0,
    last_arg = None,
    extra_kw_parameters = {},
    callback = None,
    opt_require_quota_remaining = None,
):
    if task_processor is None:
        task_processor = Task
    extra_kw_parameters['queue_name'] = queue_name
    if opt_require_quota_remaining is None \
    and 'EUD_TASK_QUOTA_REQUIRE_REMAINING' in os.environ:
        opt_require_quota_remaining = utilities.float_with_suffix(
            os.environ['EUD_TASK_QUOTA_REQUIRE_REMAINING']
        )
    if 'EUD_TASK_POLL_FREQUENCY' in os.environ:
        poll_frequency = float(os.environ['EUD_TASK_POLL_FREQUENCY'])
    else:
        poll_frequency = 1.0
    tt_task_dir = utilities.bstring(os.environ['EUD_TASK_DIR'])
    queue_dir  = b'/'.join((tt_task_dir, utilities.bstring(queue_name)))
    inbox_dir  = b'/'.join((queue_dir, b'inbox'))
    active_dir = b'/'.join((queue_dir, b'active'))
    delete_dir = b'/'.join((queue_dir, b'deleted'))
    final_dir  = b'/'.join((queue_dir, b'completed'))
    for required_dir in (inbox_dir, delete_dir, active_dir, final_dir):
        my_makedirs(required_dir)
    task_queue = TaskQueue(
        inbox_dir, active_dir, delete_dir,
        task_processor, extra_kw_parameters
    )
    utilities.random_delay(5.0)
    start_time = time.time()
    print('Worker loop starting', time.ctime(start_time))
    if opt_deadline:
        print('Deadline:', time.ctime(opt_deadline))
    if opt_stopfile:
        print('Stopfile:', opt_stopfile)
    if opt_max_idle:
        idle_deadline = start_time + opt_max_idle
        print(
            'Idle deadline (will be pushed forward on each activity):',
            time.ctime(idle_deadline)
        )
    my_active_tasks = []
    last_verbose = 0.0
    while True:
        now = time.time()
        exit_reason = None
        if opt_max_idle and now > idle_deadline:
            exit_reason = 'Reached maximum idle time'
        if opt_deadline and now > opt_deadline:
            exit_reason = 'Reached deadline'
        if opt_stopfile and os.path.exists(opt_stopfile):
            exit_reason = 'Found stop file'
        if opt_require_quota_remaining \
        and utilities.quota_remaining() < opt_require_quota_remaining:
            exit_reason = 'Remaining disk quota too low'
        if callback:
            exit_reason = callback.check_limits()
        if exit_reason:
            print('\n*** %s ***\n' %exit_reason)
            if not my_active_tasks:
                if callback:
                    callback.on_worker_exit()
                sys.exit(0)
            print('Waiting for active tasks to finish, not accepting new tasks')
            task = None
        else:
            task = task_queue.pick_task(opt_debug = opt_debug)
        if task is not None:
            task.start_time = time.time()
            print('Running task %r' %task)
            sys.stderr.flush()
            sys.stdout.flush()
            task.start_processing()
            my_active_tasks.append(task)
        still_active_tasks = []
        for index, task in enumerate(my_active_tasks):
            if opt_max_idle:
                idle_deadline = time.time() + opt_max_idle
                if opt_debug:
                    print('Moved idle deadline to', time.ctime(idle_deadline))
            if not task.finished_processing():
                still_active_tasks.append(task)
                continue
            print('Detected that task %r has finished' %task)
            end_time = time.time()
            # signal completion
            bucket_dir = b'%s/%s' %(final_dir, task.task_bucket)
            my_makedirs(bucket_dir)
            final_file = b'%s/%s.task' %(bucket_dir, task.task_id)
            f = open(final_file, 'wb')
            f.write(b'duration\t%.1f\n' %(end_time-task.start_time))
            f.write(b'waiting\t%.1f\n' %(task.start_time-task.submit_time))
            f.write(b'total\t%.1f\n' %(end_time-task.submit_time))
            for keyname, envname in [
                ('cluster',  'SLURM_CLUSTER_NAME'),
                ('job_id',   'SLURM_JOB_ID'),
                ('job_name', 'SLURM_JOB_NAME'),
                ('host',     'HOSTNAME'),
            ]:
                if envname in os.environ:
                    f.write(utilities.bstring(
                        '%s\t%s\n' %(keyname, os.environ[envname])
                    ))
            f.write(b'process\t%d\n' %os.getpid())
            f.write(b'submitted\t%.1f\n' %task.submit_time)
            f.write(b'start\t%.1f\n' %task.start_time)
            f.write(b'end\t%.1f\n' %end_time)
            f.write(b'expires\t%.1f\n' %task.expires)
            f.write(b'task_id\t%s\n' %task.task_id)
            f.write(b'bucket\t%s\n' %task.task_bucket)
            f.write(b'arg_len\t%d\n' %len(task.command))
            # TODO: 'requires' header lines are currently lost
            f.write(b'\n') # empty line to mark end of header, like in http
            f.write(b'\n'.join(task.command))
            f.write(b'\n') # final newline
            f.close()
            try:
                os.unlink(task.active_name)
            except OSError:
                print('Error: Could not remove finished task %r from active queue' %task.task_id)
            if opt_max_idle:
                idle_deadline = time.time() + opt_max_idle
                if opt_debug:
                    print('Moved idle deadline to', time.ctime(idle_deadline))
        my_active_tasks = still_active_tasks
        now = time.time()
        if opt_debug or now > last_verbose + 60.0:
            print('Tasks still active:', len(my_active_tasks))
            last_verbose = now
        if callback:
            callback.on_worker_idle()
        sys.stdout.flush()
        utilities.random_delay(
            12.0/poll_frequency,
            0.8, 1.2
        )

if __name__ == "__main__":
    main('udpf', Task)

