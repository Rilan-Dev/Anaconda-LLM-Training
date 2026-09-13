import time


class Timer:

    def __init__(self):
        self.start = time.time()

    def stop(self):

        return round(time.time() - self.start, 2)