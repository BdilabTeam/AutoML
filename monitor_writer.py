from tensorboardX import SummaryWriter


class monitor_writer:
    def __init__(self):
        self.step = 0
        self.usage_writer = SummaryWriter(comment='resource')
        self.reward_writer = SummaryWriter(comment='reward')
        self.performance_writer = SummaryWriter(comment='performance')
        self.job_writer = SummaryWriter(comment='jobs')
        self.log_writer=SummaryWriter(comment='logs')

    def write_usage(self, node, mem_usage, gpu_usage):
        self.usage_writer.add_scalar('resource/gpu_mem/' + node, mem_usage, global_step=self.step)
        self.usage_writer.add_scalar('resource/gpu/' + node, gpu_usage, global_step=self.step)

    def write_reward(self, reward):
        self.reward_writer.add_scalar('reward', reward, self.step)

    def write_pm(self, delay, loss, tuned_loss):
        self.performance_writer.add_scalar('performance/delay', delay, self.step)
        self.performance_writer.add_scalar('performance/loss', loss, self.step)
        self.performance_writer.add_scalar('performance/tuned_loss', tuned_loss, self.step)

    def write_job(self, running, finished, waiting):
        self.job_writer.add_scalar('jobs/running', running, self.step)
        self.job_writer.add_scalar('jobs/finished', finished, self.step)
        self.job_writer.add_scalar('jobs/waiting_jobs', waiting, self.step)
    # def write_log(self, temp_log):
    #     for node in temp_log['node_info']:
    #         self.log_writer.add_text('node_info', str(temp_log['node_info'][node]), self.step)
    #     for job in temp_log['job_info']:
    #         if any(phase == 'Running' for phase in temp_log['job_info'][job]['phase']):
    #             self.log_writer.add_text('job_info', str(temp_log['job_info'][job]), self.step)
    #     self.log_writer.add_text('policy', str(temp_log['policy']), self.step)

    def add_step(self):
        self.step = self.step+1

    def get_step(self):
        return self.step