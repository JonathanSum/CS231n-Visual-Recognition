class Greeting(object):
    def __init__(self,name):
        self.name = name

    def greet(self, loud=False):
        if loud == True:
            print("Hello, %s!" % self.name.upper())
        else:
            print('Hello, %s' % self.name)
g = Greeting('Monster')
g.greet()
g.greet(loud=True)