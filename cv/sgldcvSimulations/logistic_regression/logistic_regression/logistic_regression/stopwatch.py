import time

class Stopwatch:
    """Define tic() and toc() for calculating time"""
    def __init__(self):
        self.current = time.time()

    def tic(self):
        """Reset stopwatch"""
        self.current = time.time()

    def toc(self):
        """Return elapsed time"""
        elapsed = time.time() - self.current
        # Reset timer
        self.current = time.time()
        return elapsed
