import os
import time


def listdir_replays_only(path, as_path=False):
    contents = [
        dirent for dirent in os.listdir(path)
        if dirent.endswith('.roa')
        and os.path.isfile(os.path.join(path, dirent))
        ]
    if as_path:
        return [ os.path.join(path, dirent) for dirent in contents ]
    return contents


def version_to_dname(string):
    '''Convert x.x.x to xx_xx_xx'''
    return '_'.join([
        char if len(char) > 1 else '0' + char
        for char in string.split('.') if char.isdigit()
        ])


def dname_to_version(string):
    '''Convert xx_xx_xx to x.x.x'''
    return '.'.join([
        str(int(char))
        for char in string.split('_') if char.isdigit()
        ])

def ensure_directory_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)


class PlaybackTimer:
    def start(self, duration):
        '''Hit the clock'''
        self.start_time = time.time()
        self.duration = duration
        self.end_time = self.start_time +  duration

    def is_playing(self):
        '''Returns true if the replay is still playing'''
        return time.time() < self.end_time

    def seconds_elapsed(self):
        '''Returns the number of seconds elapsed now'''
        return self.seconds_elapsed_since(time.time())

    def seconds_elapsed_since(self, timestamp):
        '''Returns the number of seconds elapsed for a particular time'''
        return timestamp - self.start_time

    def seconds_remaining(self):
        '''Returns the number of seconds remaining now'''
        return self.seconds_remaining_after(time.time())

    def seconds_remaining_after(self, timestamp):
        '''Returns the number of seconds remaining for a particular time'''
        return self.end_time - timestamp