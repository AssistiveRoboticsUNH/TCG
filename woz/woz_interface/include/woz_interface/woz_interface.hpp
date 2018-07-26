#ifndef WOZINTERFACE_H
#define WOZINTERFACE_H

#include <QWidget>
#include <QImage>
#include <QPainter>
#include <QPaintDevice>
#include <QtGui>
#include <QLCDNumber>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <nao_msgs/JointAnglesWithSpeed.h>
#include <std_msgs/String.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Bool.h>
#include <naoqi_bridge_msgs/BodyPoseActionGoal.h>
#include <std_srvs/Empty.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <time.h>
#include <woz_msgs/control_states.h>
#include <woz_msgs/action_label.h>
#include <nao_msgs/JointAnglesWithSpeed.h>

#include <tf/transform_listener.h>

namespace Ui {
    class WoZInterface;
}

class WoZInterface : public QWidget {
    Q_OBJECT

public:
    std::string name;

    explicit WoZInterface(QWidget *parent = 0);

    ~WoZInterface();

    void topImageCallback(const sensor_msgs::ImageConstPtr &msg);

    void bottomImageCallback(const sensor_msgs::ImageConstPtr &msg);

    void controlCallback(const woz_msgs::control_states States);

    void actionCallback(const std_msgs::Int8 &msg);

    void UpdateImage();

    void loopRate(int loop_rates);

    std::string getTimeStamp();

    void waveNao();

    void centerGaze();

    void publishActionLabel(std::string action);

private
    Q_SLOTS:
    void on_discStimulus_clicked();

    void on_prompt_clicked();

    //void on_correctingPrompt_clicked();

    void on_reward_clicked();

    void on_failure_clicked();

    void on_startRecording_clicked();

    void on_stopRecording_clicked();

    void on_stand_clicked();

    void on_rest_clicked();

    void on_angleHead_clicked();

    void on_toggleLife_clicked();

    void on_shutDown_clicked();

    void on_start_clicked();

    void on_clock_overflow();

    void on_countDown_overflow();


protected:
    void paintEvent(QPaintEvent *event);

    void timerEvent(QTimerEvent *event);

private:
    Ui::WoZInterface *ui;
    QBasicTimer Mytimer;
    QTimer *timer;
    QTimer *cdtimer;
    QString clockTimetext;
    ros::NodeHandle n;
    ros::Publisher pub_speak, pub_pose, pub_woz_msgs,
            pub_move, pub_run, pub_label_msgs, pub_start_session;
    ros::ServiceClient client_stiff, client_record_start, client_record_stop,
            client_wakeup, life_enable, life_disable, client_rest;
    ros::Subscriber sub_woz_msgs, sub_tcam, sub_bcam, sub_action_msgs;
    QImage NaoTopImg;
    QImage NaoBottomImg;
    int count;
    int timeout;
    int shortTimeout;
    int promptLimit;
    bool recording = false;
    bool life_on = true;
    woz_msgs::control_states controlstate;
    woz_msgs::action_label actionLabel;

    tf::TransformListener listener;

};

#endif
	
