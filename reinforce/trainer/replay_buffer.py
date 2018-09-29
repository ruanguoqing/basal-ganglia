import scipy.signal


class ReplayBuffer:
    @staticmethod
    def get_cumulative_rewards(rewards, discount):
        return scipy.signal.lfilter([1.], [1., -discount], rewards[::-1])[::-1]

    def __init__(self, buffer_length):
        self.idx, self.buffer_length = 0, buffer_length
        self.trace_list_by_iter = [None for _ in range(self.buffer_length)]

    def absorb_trace(self, trace_list):
        self.trace_list_by_iter[self.idx] = trace_list
        self.idx = (self.idx+1) % self.buffer_length

    def emit_trace_as_grouped(self, discount):
        s_list, a_list, r_list, cum_r_list = [], [], [], []
        for trace_list in self.trace_list_by_iter:
            for trace in trace_list:
                s_eps, a_eps, r_eps, _ = list(zip(*trace))
                s_list, a_list, r_list = s_list+list(s_eps), a_list+list(a_eps), r_list+list(r_eps)
                cum_r_list += list(self.get_cumulative_rewards(r_eps, discount))
        return s_list, a_list, r_list, cum_r_list
