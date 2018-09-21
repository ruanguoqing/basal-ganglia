import gym


class Game:
    @staticmethod
    def parse_xput(xput):
        xput_dict = {gym.spaces.Box: ('Box', lambda x: x.shape[0]),
                     gym.spaces.Discrete: ('Discrete', lambda x: x.n)}
        xput_type, xput_lam = xput_dict[type(xput)]
        return xput_type, xput_lam(xput)

    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state_info = self.parse_xput(self.env.observation_space)
        self.action_info = self.parse_xput(self.env.action_space)

    def do_rollout(self, policy, reset=True, t_eps=10**3):
        if reset: s = self.env.reset()
        trace = []
        for i_step in range(t_eps):
            a = policy(s)
            s_new, r, is_done, info = self.env.step(a)
            trace.append((s, a, r, s_new))
            s = s_new
            if is_done: break
        return trace

    def play_episodes(self, policy, n_eps=1):
        return [self.do_rollout(policy) for _ in range(n_eps)]
