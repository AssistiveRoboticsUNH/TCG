#include <ros/ros.h>
#include <string>
#include <woz_msgs/control_states.h>
#include <woz_msgs/action_label.h>
#include <iostream>
#include <nao_msgs/JointAnglesWithSpeed.h>
#include <naoqi_bridge_msgs/BodyPoseActionGoal.h>

woz_msgs::control_states states;

/* Gets the current timestamp whenever it is called */
std::string getTimeStamp() {
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, 80, "%Y-%m-%d-%H-%M-%S", timeinfo);
    std::string str(buffer);
    return str;
}

void cb(const woz_msgs::control_states States){
	states = States;
}

int main(int argc, char ** argv) {
    ros::init(argc, argv, "woz_actions_point");
    ros::NodeHandle n;

    ros::Subscriber sub_control = n.subscribe("/woz_msgs", 100, cb);
    ros::Publisher pub_label_msgs = n.advertise<woz_msgs::action_label>("/action_started", 100);
    ros::Publisher pub_control = n.advertise<woz_msgs::control_states>("/woz_msgs", 100);
    ros::Publisher pub_move = n.advertise<nao_msgs::JointAnglesWithSpeed>("/joint_angles", 100);
    ros::Publisher pub_pose = n.advertise<naoqi_bridge_msgs::BodyPoseActionGoal>("/body_pose/goal",
                                                                                 100);

    ros::Rate loop_rate(15);

    std::string action;
    nao_msgs::JointAnglesWithSpeed rh, rsp, rsr, rw, rer, rey;

    rsp.joint_names.push_back("RShoulderPitch");
    rsp.joint_angles.push_back(0);

    rsr.joint_names.push_back("RShoulderRoll");
    rsr.joint_angles.push_back(0);

    rh.joint_names.push_back("RHand");
    rh.joint_angles.push_back(0);

    rw.joint_names.push_back("RWristYaw");
    rw.joint_angles.push_back(0);

    rer.joint_names.push_back("RElbowRoll");
    rer.joint_angles.push_back(0);

    rey.joint_names.push_back("RElbowYaw");
    rey.joint_angles.push_back(0);

    naoqi_bridge_msgs::BodyPoseActionGoal pose;
    int i;

    while (ros::ok()) {
        ros::spinOnce();
        if (!states.startPointing && !states.startPointing2 && !states.startPointing3) {
            ros::spinOnce();
        } else if (states.shutdown) {
            ROS_INFO("SHUTTING DOWN POINTING");
            ros::shutdown();
        } else if (states.startPointing) {
            action = "sd";
            rsp.joint_angles[0] = 0.7;
            rsp.speed = 0.25;
            rw.joint_angles[0] = 1.3;
            rw.speed = 0.25;

            pub_move.publish(rw);
            pub_move.publish(rsp);

            ros::Duration(1).sleep();

            rsp.joint_angles[0] = 1.0;
            rsp.speed = 0.25;

            rsr.joint_angles[0] = 0.35;
            rsr.speed = 0.25;

            rh.joint_angles[0] = 1;
            rh.speed = 0.25;

            pub_move.publish(rsp);
            pub_move.publish(rsr);
            pub_move.publish(rh);
            ros::Duration(1.5).sleep();

            rsp.joint_angles[0] = 1.5;
            rsp.speed = 0.25;

            rsr.joint_angles[0] = -0.16;
            rsr.speed = 0.25;

            rh.joint_angles[0] = 0.3;
            rh.speed = 0.25;

            rw.joint_angles[0] = 0.09;
            rw.speed = 0.25;

            pub_move.publish(rsp);
            pub_move.publish(rsr);
            pub_move.publish(rh);
            pub_move.publish(rw);
            loop_rate.sleep();

            states.startPointing = false;
            pub_control.publish(states);
        } else if (states.startPointing2) {
            action = "prompt";
            rsp.joint_angles[0] = 0.9;
            rsp.speed = 0.25;

            rsr.joint_angles[0] = 0.35;
            rsr.speed = 0.25;

            rh.joint_angles[0] = 1;
            rh.speed = 0.25;

            pub_move.publish(rsp);
            pub_move.publish(rsr);
            pub_move.publish(rh);

            ros::Duration(2).sleep();

            rsp.joint_angles[0] = 1.5;
            rsp.speed = 0.25;

            rsr.joint_angles[0] = -0.16;
            rsr.speed = 0.25;

            rh.joint_angles[0] = 0.3;
            rh.speed = 0.25;

            pub_move.publish(rsp);
            pub_move.publish(rsr);
            pub_move.publish(rh);
            loop_rate.sleep();

            states.startPointing2 = false;
            pub_control.publish(states);
        } else if (states.startPointing3) {
            action = "reward";
            rsp.joint_angles[0] = 0.5;
            rsp.speed = 0.25;

            rsr.joint_angles[0] = 0.35;
            rsr.speed = 0.25;

            rh.joint_angles[0] = 0.1;
            rh.speed = 0.25;

            rer.joint_angles[0] = 0.8;
            rer.speed = 0.4;

            pub_move.publish(rsp);
            pub_move.publish(rsr);
            pub_move.publish(rh);
            pub_move.publish(rer);

            ros::Duration(1.5).sleep();

            rsp.joint_angles[0] = 1.6;
            rsp.speed = 0.4;

            rer.joint_angles[0] = 1.1;
            rer.speed = 0.4;

            pub_move.publish(rsp);
            pub_move.publish(rer);

            ros::Duration(0.5).sleep();

            rsp.joint_angles[0] = 0.5;
            rsp.speed = 0.4;

            pub_move.publish(rsp);

            ros::Duration(0.5).sleep();

            rsp.joint_angles[0] = 1.5;
            rsp.speed = 0.25;

            rsr.joint_angles[0] = -0.16;
            rsr.speed = 0.25;

            rh.joint_angles[0] = 0.3;
            rh.speed = 0.25;

            rer.joint_angles[0] = 0.41;
            rer.speed = 0.25;

            pub_move.publish(rsp);
            pub_move.publish(rsr);
            pub_move.publish(rh);
            pub_move.publish(rer);
            loop_rate.sleep();

            states.startPointing3 = false;
            pub_control.publish(states);
        }
    }
    woz_msgs::action_label actionLabel;
    actionLabel.timestamp = getTimeStamp();
    actionLabel.action = action + " ended";
    pub_label_msgs.publish(actionLabel);
    return 0;
}
