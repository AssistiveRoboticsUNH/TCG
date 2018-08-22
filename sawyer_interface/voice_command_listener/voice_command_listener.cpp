#include <thread>
#include <chrono>
#include <atomic>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Int8.h"

bool busy;
ros::Publisher pubState;

void commandParser(const std_msgs::StringConstPtr& msg) {
    ROS_INFO("%s", msg->data.c_str());
	if (busy == false && (msg->data == "inspection started" or msg->data == "started")) {
	    busy = true;
	    std_msgs::Int8 out;
	    out.data = 0;
	    pubState.publish(out);
	}
	else if (busy == true && (msg->data == "inspection completed" or msg->data == "completed")) {
	    busy = false;
	    std_msgs::Int8 out;
	    out.data = 1;
	    pubState.publish(out);
	}
}

int main(int argc, char **argv) {
	busy = false;
	ros::init(argc, argv, "voice_command_listener");
	ros::NodeHandle n;
	ROS_INFO("Node initialized.");
	ros::Subscriber subRecognizer = n.subscribe("/recognizer/output", 20, commandParser);
	pubState = n.advertise<std_msgs::Int8>("/human_action", 10);
	ros::spin();
	return 0;
}


