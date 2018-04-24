import time

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