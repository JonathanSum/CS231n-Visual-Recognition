class MultiplyGate(object):
    def forward(self, x, y):
        z = x * y
        self.x = x  # must keep these around
        self.y = y
        return z

    def backward(self, dz):
        # dx is y times dz
        dx = self.y * dz  # [dz/dx* dL/dz]
        dy = self.x * dz
        return [dx, dy]
