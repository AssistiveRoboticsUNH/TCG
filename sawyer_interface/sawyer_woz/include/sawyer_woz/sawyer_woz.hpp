#ifndef SAWYER_WOZINTERFACE_H
#define SAWYER_WOZINTERFACE_H

#include <QWidget>
#include <QImage>
#include <QPainter>
#include <QPaintDevice>
#include <QtGui>
#include <QLCDNumber>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <std_msgs/String.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Bool.h>
#include <std_srvs/Empty.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <time.h>

#include <tf/transform_listener.h>

namespace Ui {
    class SawyerWoZInterface;
}

class SawyerWoZInterface : public QWidget {
    Q_OBJECT

public:
    explicit SawyerWoZInterface(QWidget *parent = 0);

    ~SawyerWoZInterface();

    void controlCallback(const std_msgs::Int8 &msg);

    void actionCallback(const std_msgs::Int8 &msg);

    void UpdateImage();

private
    Q_SLOTS:
    void on_pickItem_clicked();

    void on_placeDQA_clicked();

    void on_placeSQA_clicked();

    void on_placeBox_clicked();

    void on_startRecording_clicked();

    void on_stopRecording_clicked();

    void on_moveToStart_clicked();

    void on_run_clicked();

    void on_countDown_overflow();


protected:
    void timerEvent(QTimerEvent *event);

private:
    Ui::SawyerWoZInterface *ui;
    QBasicTimer Mytimer;
    QTimer *timer;
    QTimer *cdtimer;
    ros::NodeHandle n;
    ros::Publisher pub_sawyer_woz_msgs, pub_run;
    ros::ServiceClient client_record_start, client_record_stop;
    ros::Subscriber sub_sawyer_msgs, sub_action_msgs;
    int count;
    int longTimeout;
    int shortTimeout;
    int itemLimit;
    bool recording = false;
    tf::TransformListener listener;

};

#endif
	
