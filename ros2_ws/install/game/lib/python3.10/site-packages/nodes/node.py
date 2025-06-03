import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.parameter import Parameter
from std_msgs.msg import String
from threading import Thread
from functools import partial
from agents import *
from configs import *
import time

class Node(Node):
    MAX_CACHE_NUMS = 20
    def __init__(self):
        super().__init__('agent')
        
        self.declare_parameter('node_id', Parameter.Type.INTEGER)
        self.declare_parameter('config', Parameter.Type.STRING)
        self.node_id = self.get_parameter('node_id').get_parameter_value().integer_value
        game, index = self.get_parameter('config').get_parameter_value().string_value.split('@')
        self.config = CONFIG_MAP[game][index]

        self.adjacency_matrix = self.config['adjacency_matrix']
        self.config['agent_config']['config_index'] = index
        self.config['agent_config']['node_id'] = self.node_id
        self.agent = Agent(self.config['agent_config'])
        
        topic_name = f"{game}_{index}"
        self.agent_publisher = self.create_publisher(String, f'/{topic_name}/agent_{self.node_id}', self.MAX_CACHE_NUMS)
        self.agent_subscriber_list = []
        self.adj_agent_nums = 0
        for i, edge_value in enumerate(self.adjacency_matrix[self.node_id]):
            if i == self.node_id:
                continue
            if np.fabs(edge_value) > 1e-4:
                subscription = self.create_subscription(String, f'/{topic_name}/agent_{i}',  partial(self.receieve_callback, i), self.MAX_CACHE_NUMS)
                self.agent_subscriber_list.append(subscription)
                self.adj_agent_nums += 1
        self.sync_publisher = self.create_publisher(String, f'/{topic_name}/sync_{self.node_id}', 1)
        self.sync_subscribe = self.create_subscription(String, f'/{topic_name}/sync_flag', self.sync_callback, 1)
        self.last_sync_flag = None
        self.adj_agent_set = set()
        
        self.is_alive = True
        
    def work(self):
        self.agent.update()
    
    def get_pub_msg(self):
        return self.agent.send_msg()
    
    def receieve_callback(self, id, msg):
        if id not in self.adj_agent_set:
            self.adj_agent_set.add(id)
            self.agent.receieve_msg(id, msg.data)
        if len(self.adj_agent_set) == self.adj_agent_nums:
            self.adj_agent_set.clear()
            syn_msg = String()
            syn_msg.data = str("ok")
            self.sync_publisher.publish(syn_msg)
        
    def sync_callback(self, msg):
        sync_flag = int(msg.data)
        if sync_flag == -1:
            self.agent.done()
            self.stop_node()
        if sync_flag is None or sync_flag != self.last_sync_flag:
            self.work()
            pub_msg = String()
            pub_msg.data = self.get_pub_msg()
            self.agent_publisher.publish(pub_msg)
            
    def stop_node(self):
        self.get_logger().info('Stopping node...')
        time.sleep(1)
        self.is_alive = False

def main(args=None):
    rclpy.init(args=args)
    node = Node()
    while node.is_alive:
        rclpy.spin_once(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
