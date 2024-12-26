import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import Log
import time
import math
class Mapping(Node):
    def __init__(self):
        super().__init__("mapping_node")
        self.is_moving = False
        self.is_finish = False
        self.sub_map = self.create_subscription(OccupancyGrid,'/map',self.callback_mapping,10)
        self.pub_goal = self.create_publisher(PoseStamped,'/goal_pose',10)
        self.sub_status = self.create_subscription(Log,'/rosout',self.callback_goal,10)
    def callback_goal(self,msg):
        if 'Goal succeeded' in msg.msg:
            #self.get_logger().info(msg.msg)
            self.is_moving = False
    def callback_mapping(self,msg):
        if self.is_moving or self.is_finish:
            return
        mapping = msg
        width = mapping.info.width
        height = mapping.info.height
        origin_x = -round(mapping.info.origin.position.x,3)
        origin_y = -round(mapping.info.origin.position.y,3)
        per_pixel = round(mapping.info.resolution,3)
        np_map = np.array(mapping.data).reshape(height,width)
        init_pose_x = int(origin_x/per_pixel)
        init_pose_y = int(origin_y/per_pixel)
        goal_pose = self.goal_pose_detection(np_map,(init_pose_y,init_pose_x))
        if goal_pose[0] == -1:
            goal = PoseStamped()
            goal.header.frame_id = 'map'
            goal.pose.position.x = 0
            goal.pose.position.y = 0
            goal.pose.orientation.z = 0.0
            goal.pose.orientation.w = 0.0
            goal.header.stamp = self.get_clock().now().to_msg()
            #for _ in range(5):
            self.pub_goal.publish(goal)
            self.is_finish = True
            self.get_logger().info('go to init pose')
            return
        self.get_logger().info(f"init pose: {init_pose_x}, {init_pose_y}")
        #print('origin:',origin_x,origin_y)
        local_map_x = goal_pose[0] - init_pose_x
        local_map_y = goal_pose[1] - init_pose_y
        goal_x = local_map_x * per_pixel
        goal_y = local_map_y * per_pixel
        #print('local map', local_map_x, local_map_y)
        self.get_logger().info(f"goal: {goal_x}, {goal_y}")
        #print("global:",global_goal_x, global_goal_y)
        #print("local:",local_map)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.orientation.z = 0.0
        goal.pose.orientation.w = 0.0
        goal.header.stamp = self.get_clock().now().to_msg()
        #for _ in range(5):
        self.pub_goal.publish(goal)
        self.get_logger().info("publish goal")
        self.is_moving = True
        # for raw in np_map:
        #     print(raw)
        #time.sleep(1)
    def distance(self,p1,p2):
        return ( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
    def goal_pose_detection(self,map,home):
        home_map = map
        height = home_map.shape[0]
        width = home_map.shape[1]
        init_location = home
        free_zone_idx = np.argwhere(home_map == 0)
        dists = []
        poses = []
        for pose in free_zone_idx:
            is_insert = False
            is_in = [True,True,True,True]    # 위쪽부터 시계 방향으로 검사할 것인가 판단
            if pose[0]==0: # 위쪽 좌표 검사 여부
                is_in[0] = False
            if pose[1]==width-1: # 우측 좌표 검사 여부
                is_in[1] = False
            if pose[0]==height-1: # 아래쪽 좌표 검사 여부
                is_in[2] = False
            if pose[1]==0: # 좌측 좌표 검사 여부
                is_in[3] = False
            if is_in[0] and home_map[pose[0]-1][pose[1]] == -1:
                is_insert = True
            if is_in[1] and home_map[pose[0]][pose[1]+1] == -1:
                is_insert = True
            if is_in[2] and home_map[pose[0]+1][pose[1]] == -1:
                is_insert = True
            if is_in[3] and home_map[pose[0]][pose[1]-1] == -1:
                is_insert = True
            count = 0
            if is_in[0] and home_map[pose[0]-1][pose[1]] == 0:
                count +=1
            if is_in[1] and home_map[pose[0]][pose[1]+1] == 0:
                count +=1
            if is_in[2] and home_map[pose[0]+1][pose[1]] == 0:
                count +=1
            if is_in[3] and home_map[pose[0]][pose[1]-1] == 0:
                count +=1
            if is_insert and count >=3:
                dists.append(self.distance(pose,init_location))
                i = [pose[1],pose[0]]
                poses.append(i)
        if len(poses) == 0:
            self.get_logger().info('finish mapping!')
            goal_pose = [-1,-1]
            return goal_pose
        goal_pose = poses[dists.index(max(dists))]
        self.get_logger().info(f"goal_pose: {goal_pose}")
        return goal_pose
#goal_pose_detection(home_map,(6,5))
if __name__ == '__main__':
    #print(np.array(m).reshape(7,6))
    rclpy.init()
    node = Mapping()
    rclpy.spin(node)
    rclpy.shutdown()
    node.destroy_node()