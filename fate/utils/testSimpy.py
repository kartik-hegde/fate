import simpy

# basic example
def basic_example():

    def clock(env, name, tick):
        while True:
            print(name, env.now)
            yield env.timeout(tick)

    env = simpy.Environment()
    env.process(clock(env, 'fast', 0.5))
    env.process(clock(env, 'slow', 1))
    env.run(until=2)
    
if __name__=='__main__':
    basic_example()
