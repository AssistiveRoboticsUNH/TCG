/*
sawyer_woz.cpp
Estuardo Carpio-Mazariegos
Aug 2018
*/
#include "../include/sawyer_woz/sawyer_woz.hpp"
#include "../include/sawyer_woz/sawyer_ui_woz.hpp"
#include <string.h>



/* Class Constructor: Initializes all of the QT slots and widgets, and initializes all of the subscribers,
 * publishers, and services */
SawyerWoZInterface::SawyerWoZInterface(QWidget *parent) : QWidget(parent), ui(new Ui::SawyerWoZInterface) {

    /* Sets up UI */
    cdtimer = new QTimer(this);
    connect(cdtimer, SIGNAL(timeout()), this, SLOT(on_countDown_overflow()));
    Mytimer.start(100, this);
    cdtimer->start(1000);
    ui->setupUi(this);

    /* Sets up ROS */
    ros::start();

    //sets count to 0 so program can go through ros::spinOnce 10 times to solve issue with seg fault
    count = 0;
    longTimeout = 2;
    shortTimeout = 1;
    itemLimit = 6;
    humanReady = true;
    recording = false;

    //advertises state status
    pub_sawyer_woz_msgs = n.advertise<std_msgs::Int8>("/sawyer_woz_action", 100);
    pub_run = n.advertise<std_msgs::Bool>("/run_sawyer_auto", 100);

    // service client to start rosbag
    client_record_start = n.serviceClient<std_srvs::Empty>("/data_logger/start");
    // service client to stop rosbag
    client_record_stop = n.serviceClient<std_srvs::Empty>("/data_logger/stop");

    // subscriber to get state status
    sub_sawyer_msgs = n.subscribe("sawyer_msgs", 100, &SawyerWoZInterface::controlCallback, this);

    // subscriber to get external actions executed
    sub_action_msgs = n.subscribe("tcg_msgs", 1, &SawyerWoZInterface::actionCallback, this);

    sub_human_msgs = n.subscribe("human_action", 1, &SawyerWoZInterface::humanCallback, this);
}

/* Destructor: Frees space in memory where ui was allocated */
SawyerWoZInterface::~SawyerWoZInterface() {
    delete ui;
}

/* Updates response countdown if necessary */
void SawyerWoZInterface::on_countDown_overflow() {
    int time = ui->countDown->intValue() - 1;
    if (time >= 1) {
        ui->countDown->display(time);
    } else if (time == 0 && humanReady == false) {
        ui->countDown->display(-1);
    }
}

/* Activates an action if it is called via rostopic */
void SawyerWoZInterface::actionCallback(const std_msgs::Int8& msg) {
    int act = msg.data;
    if (act == 0) {
        on_pickItem_clicked();
    } else if (act == 1) {
        on_placeDQA_clicked();
    } else if (act == 2) {
        on_placeSQA_clicked();
    } else if (act == 3) {
        on_placeBox_clicked();
    }
}

void SawyerWoZInterface::on_startRecording_clicked() {
    std_srvs::Empty msg;
    client_record_start.call(msg);
    recording = true;
}

void SawyerWoZInterface::on_stopRecording_clicked() {
    std_srvs::Empty stop;
    if (recording) {
        client_record_stop.call(stop);
    }
    recording = false;
}

void SawyerWoZInterface::on_pickItem_clicked() {
    int op = ui->itemLimit->intValue();
    if (op > 0) {
        ui->itemLimit->display(op - 1);
        std_msgs::Int8 msg;
        msg.data = op;
        pub_sawyer_woz_msgs.publish(msg);
    }
}

void SawyerWoZInterface::on_placeDQA_clicked() {
    std_msgs::Int8 msg;
    msg.data = 7;
    pub_sawyer_woz_msgs.publish(msg);
}

void SawyerWoZInterface::on_placeSQA_clicked() {
    std_msgs::Int8 msg;
    msg.data = 8;
    pub_sawyer_woz_msgs.publish(msg);
}

void SawyerWoZInterface::on_placeBox_clicked() {
    std_msgs::Int8 msg;
    msg.data = 9;
    pub_sawyer_woz_msgs.publish(msg);
}

void SawyerWoZInterface::timerEvent(QTimerEvent*) {
    update();
    UpdateImage();
}

void SawyerWoZInterface::UpdateImage() {
    ros::spinOnce();
}

void SawyerWoZInterface::controlCallback(const std_msgs::Int8& msg) {
    ui->countDown->display(longTimeout);
}

void SawyerWoZInterface::humanCallback(const std_msgs::Int8& msg) {
    if (msg.data == 0) {
        humanReady = false;
    } else {
        humanReady = true;
    }
}

void SawyerWoZInterface::on_moveToStart_clicked() {
    ui->itemLimit->display(itemLimit);
    std_msgs::Int8 msg;
    msg.data = 0;
    pub_sawyer_woz_msgs.publish(msg);
}

void SawyerWoZInterface::on_run_clicked() {
    ui->itemLimit->display(itemLimit);
    std_msgs::Bool msg = std_msgs::Bool();
    pub_run.publish(msg);
}