import geppy as gep
from deap import creator, base, tools
import numpy as np
import random
import operator 
import warnings
warnings.filterwarnings("ignore")

class GeneticProgram:
    def __init__(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        
        self.toolbox = gep.Toolbox()
        self.pset = None
        self.setup()

    def f(self, x):
        """Objective function"""
        return -2 * x ** 2 + 11 * x + 13

    def generate_dataset(self, n_cases=100):
        X = np.random.uniform(-10, 10, size=n_cases)
        Y = self.f(X) + np.random.normal(size=n_cases)
        return X, Y

    def protected_div(self, x1, x2):
        if abs(x2) < 1e-6:
            return 1
        return x1 / x2

    def setup(self):
        # 定义基本的算术运算符和函数
        self.pset = gep.PrimitiveSet('Main', input_names=['x'])
        self.pset.add_function(operator.add, 2)
        self.pset.add_function(operator.sub, 2)
        self.pset.add_function(operator.mul, 2)
        self.pset.add_function(self.protected_div, 2)
        self.pset.add_ephemeral_terminal(name='enc', gen=lambda: random.randint(-10, 10))

        creator.create("FitnessMin", base.Fitness, weights=(-1,))
        creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

        h = 7
        n_genes = 2
        self.toolbox.register('gene_gen', gep.Gene, pset=self.pset, head_length=h)
        self.toolbox.register('individual', creator.Individual, gene_gen=self.toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register('compile', gep.compile_, pset=self.pset)
        
    def evaluate(self, individual, X, Y):
        func = self.toolbox.compile(individual)
        Yp = np.array(list(map(func, X)))
        return np.mean(np.abs(Y - Yp)),

    def init_toolbox_with_evaluate(self, X, Y):
        self.toolbox.register('evaluate', self.evaluate, X=X, Y=Y)
                              
    def run(self, n_pop=100, n_gen=100):
        # 首先生成数据集
        X, Y = self.generate_dataset(n_cases=100)
        
        # 然后初始化针对具体数据评价函数的工具箱设置
        self.init_toolbox_with_evaluate(X, Y)
        
        # 注册遗传算法的其它操作
        self.toolbox.register('select', tools.selTournament, tournsize=3)
        self.toolbox.register('mut_uniform', gep.mutate_uniform, pset=self.pset, ind_pb=0.05, pb=1)
        self.toolbox.register('mut_invert', gep.invert, pb=0.1)
        self.toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
        self.toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
        self.toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.2)
        self.toolbox.register('cx_1p', gep.crossover_one_point, pb=0.2)
        self.toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
        self.toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)

        self.toolbox.register('mut_ephemeral', gep.mutate_uniform_ephemeral, ind_pb='1p')  # 1p: expected one point mutation in an individual
        self.toolbox.pbs['mut_ephemeral'] = 1  # 也可以这样来设置概率
        
        # 统计
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # 初始化种群
        pop = self.toolbox.population(n=n_pop)
        
        # 初始化精英个体的记录器
        hof = tools.HallOfFame(3)
        
        # 运行遗传算法
        pop, log = gep.gep_simple(pop, self.toolbox, n_generations=n_gen, n_elites=1, stats=stats, hall_of_fame=hof, verbose=True)
        
        return pop, log, hof

'''
# 生成实例并运行
gp = GeneticProgram(seed=42)
pop, log, hof = gp.run(n_pop=100, n_gen=30)  # 为了测试，可以先设定较少的世代数
'''