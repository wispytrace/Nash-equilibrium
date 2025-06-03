import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.parameter import Parameter
from std_msgs.msg import String
from threading import Thread
from functools import partial
from configs import *
import time

class Sync(Node):
    MAX_CACHE_NUMS = 10
    CACHE_TIME = 1
    EOF_FLAG = -1
    PROCESS_BAR_INTERVAL = 2000
    def __init__(self):
        super().__init__('sync')
        self.declare_parameter('node_nums', Parameter.Type.INTEGER)
        self.declare_parameter('config', Parameter.Type.STRING)
        self.node_nums = self.get_parameter('node_nums').get_parameter_value().integer_value
        game, index = self.get_parameter('config').get_parameter_value().string_value.split('@')
        self.config = CONFIG_MAP[game][index]
        self.epochs = self.config['epochs']

        topic_name = f"{game}_{index}"
        self.sync_subscriber_list = []
        for i in range(self.node_nums):
            subscription = self.create_subscription(String, f'/{topic_name}/sync_{i}',  partial(self.sync_callback, i), self.MAX_CACHE_NUMS)
            self.sync_subscriber_list.append(subscription)
        self.sync_publisher = self.create_publisher(String, f'/{topic_name}/sync_flag', 1)

        self.sync_flag = 0
        self.counts = 0
        self.agent_ready_set = set()
        
        self.start_time = time.time()
        self.time_estimate = time.time()
        self.start()
        
        self.is_alive = True
        
    def start(self):
        time.sleep(self.CACHE_TIME)
        self.begin = time.time()
        msg = String()
        msg.data = str(self.sync_flag)
        self.sync_publisher.publish(msg)

    def sync_callback(self, id, msg):
        self.agent_ready_set.add(id)
        if len(self.agent_ready_set) == self.node_nums:
            self.agent_ready_set.clear()
            self.sync_flag = 1 - self.sync_flag
                        
            sync_msg = String()
            sync_msg.data = str(self.sync_flag)
            self.sync_publisher.publish(sync_msg)
            
            self.counts += 1
            if self.counts % self.PROCESS_BAR_INTERVAL == 0:
                self.publish_process_bar()
        
            if self.counts > self.epochs:
                print(self.counts)
                msg.data = str(self.EOF_FLAG)
                self.sync_publisher.publish(msg)
                time.sleep(self.CACHE_TIME)
                self.stop_node()  
                

    def stop_node(self):
        self.get_logger().info('Stopping node...')
        time.sleep(1)
        self.is_alive = False

    def seconds_to_hms_string(self, seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{round(seconds,2):04}"
    
    def publish_process_bar(self):
        now_time = time.time()
        interval = now_time - self.time_estimate
        total_time = (self.epochs / self.PROCESS_BAR_INTERVAL) * interval
        used_time = now_time - self.start_time
        total_time = self.seconds_to_hms_string(total_time)
        used_time = self.seconds_to_hms_string(used_time)
        square_num = int(int(self.counts / self.epochs * 100) / 10)
        square = "■" * square_num
        blank = " " * (10 -square_num)
        bar = f"|{square}{blank}|"
        ratio = round(self.counts / self.epochs * 100, 3) 
        message = f"{ratio:5}% {bar} {str(self.counts) + '/' + str(self.epochs):15} [{used_time}<{total_time}, {round(interval,2)}s/{self.PROCESS_BAR_INTERVAL}its]"
        # self.get_logger().info('\033[2J\033[;H')
        self.get_logger().info(message)
        self.time_estimate = now_time


def main(args=None):
    rclpy.init(args=args)
    sync = Sync()
    while sync.is_alive:
        rclpy.spin_once(sync)
    sync.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
