import gym


class Game:
    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def do_rollout(self, policy, reset=True, t_eps=10**4):
        if reset: s = self.env.reset()
        trace = []
        for i_step in range(t_eps):
            a = policy(s)
            s_new, r, is_done, info = self.env.step(a)
            trace.append((s, a, r, s_new))
            s = s_new
            if is_done: break
        return trace

    def play_episodes(self, policy, n_eps):
        return [self.do_rollout(policy) for _ in range(n_eps)]
