import numpy as np 
import math
import matplotlib.pyplot as plt 

class Bandit:
    def __init__(self,m):
        self.m = m
        self.mean = 0
        self.N = 0
    

    def pull(self):
        if np.random.uniform(0,1) < self.m:
            return 1
        else:
            return 0

    def update(self,x):
        self.N += 1
        self.mean =  (1-1.0/self.N)*self.mean + (1/self.N)*x

def ucb(mean,n,nj):
    return mean + np.sqrt(2*np.log(n)/nj)

def run_experiment(m1,m2,m3,eps,N):
        bandits = [Bandit(m1),Bandit(m2),Bandit(m3)]
        mean_list = []
        total_plays = 0

        for j in range(len(bandits)):
            x = bandits[j].pull()
            total_plays += 1
            bandits[j].update(x)

        data = np.empty(N)
        
        for i in range(N):
            #upper_confidence_bound
            p =  np.random.uniform(0,1)
            if p < eps:
                j = np.random.choice(3)
            else:
                j = np.argmax([ucb(b.mean,total_plays,b.N) for b in bandits])           
            x = bandits[j].pull()
            total_plays += 1
            bandits[j].update(x)
            data[i] = x
        
        cumulative_average = np.cumsum(data)/(np.arange(N)+1)
        plt.plot(cumulative_average)
       # plt.plot(np.ones(N)*m1)
        #plt.plot(np.ones(N)*m2)
        #plt.plot(np.ones(N)*m3)
       # plt.xscale('log')
        plt.show()
        for b in bandits:
            mean_list.append(b.mean)
            print(b.mean)
        return cumulative_average
        print(mean_list)


if __name__ == '__main__':
    c1 = run_experiment(0.2,0.45,0.1,0.5,100000)
    c2 = run_experiment(0.2,0.45,0.1,0.25,100000)
    c3 = run_experiment(0.2,0.45,0.1,0.85,100000)
    plt.plot(range(len(c1)),c1)
    plt.plot(range(len(c2)),c2)
    plt.plot(range(len(c3)),c3)
    plt.plot(c1,c2,c3)
    plt.legend(["c1",
  "c2","c3"], loc ="lower right")
    plt.show()
    # plt.xscale('log')
    # plt.show()