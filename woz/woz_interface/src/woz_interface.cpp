/*
woz_interface.cpp
Estuardo Carpio-Mazariegos
Jun 2017

Based on Madison Clark-Turner's asdinterface.cpp
*/
#include "../include/woz_interface/woz_interface.hpp"
#include "../include/woz_interface/ui_woz_interface.hpp"
#include <string.h>



/* Class Constructor: Initializes all of the QT slots and widgets, and initializes all of the subscribers,
 * publishers, and services */
WoZInterface::WoZInterface(QWidget *parent) : QWidget(parent), ui(new Ui::WoZInterface) {

    /* Sets up UI */
    timer = new QTimer(this);
    cdtimer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(on_clock_overflow()));
    connect(cdtimer, SIGNAL(timeout()), this, SLOT(on_countDown_overflow()));
    Mytimer.start(100, this);
    timer->start(100);
    cdtimer->start(1000);
    ui->setupUi(this);

    /* Sets up ROS */
    ros::start();

    //sets count to 0 so program can go through ros::spinOnce 10 times to solve issue with seg fault
    count = 0;
    timeout = 9;
    shortTimeout = 4;
    promptLimit = 5;

    //advertises state status
    pub_woz_msgs = n.advertise<woz_msgs::control_states>("/woz_msgs", 100);
    pub_label_msgs = n.advertise<woz_msgs::action_label>("/action_started", 100);
    pub_move = n.advertise<nao_msgs::JointAnglesWithSpeed>("/joint_angles", 100);
    pub_pose = n.advertise<naoqi_bridge_msgs::BodyPoseActionGoal>("/body_pose/goal", 100);
    pub_speak = n.advertise<std_msgs::String>("/speech", 100);
    pub_start_session = n.advertise<std_msgs::Int8>("/start_request", 1);
    //publisher to start automated intervention
    pub_run = n.advertise<std_msgs::Bool>("/run_woz_auto", 100);

    //service client to stiffen nao
    client_stiff = n.serviceClient<std_srvs::Empty>("/body_stiffness/enable", 100);
    // service client to start rosbag
    client_record_start = n.serviceClient<std_srvs::Empty>("/data_logger/start");
    // service client to stop rosbag
    client_record_stop = n.serviceClient<std_srvs::Empty>("/data_logger/stop");
    // service client to wake nao up
    client_wakeup = n.serviceClient<std_srvs::Empty>("/wakeup");
    // service client to place nao in rest-mode
    client_rest = n.serviceClient<std_srvs::Empty>("/rest");
    // service client to enable autonomous life
    life_enable = n.serviceClient<std_srvs::Empty>("/life/enable");
    // service client to disable autonomous life
    life_disable = n.serviceClient<std_srvs::Empty>("/life/disable");

    sub_tcam = n.subscribe<sensor_msgs::Image>("top_camera", 1,
                                               &WoZInterface::topImageCallback, this);
    sub_bcam = n.subscribe<sensor_msgs::Image>("bottom_camera", 1,
                                               &WoZInterface::bottomImageCallback, this);
    // subscriber to get state status
    sub_woz_msgs = n.subscribe("woz_msgs", 100, &WoZInterface::controlCallback, this);

    // subscriber to get external actions executed
    sub_action_msgs = n.subscribe("action_msgs", 1, &WoZInterface::actionCallback, this);
}

/* Destructor: Frees space in memory where ui was allocated */
WoZInterface::~WoZInterface() {
    delete ui;
}

/* When clock is overflowed, update time */
void WoZInterface::on_clock_overflow() {
    QTime time = QTime::currentTime();
    QString text = time.toString("hh:mm:ss");
    ui->clock->display(text);
}

/* Updates response countdown if necessary */
void WoZInterface::on_countDown_overflow() {
    int time = ui->countDown->intValue() - 1;
    if (time >= 0) {
        ui->countDown->display(time);
    }
}

/* Updates the displayed Image on UI */
void WoZInterface::UpdateImage() {
    //spins to call callback to update image information
    ros::spinOnce();
}

/* Paints the camera image and clock to the UI */
void WoZInterface::paintEvent(QPaintEvent *event) {
    QPainter myPainter(this);
    QPointF p(20, 330);
    QPointF p2(20, 590);
    // first few frames are corrupted, so cannot draw image until it gets at least 10 frames
    if (count >= 10) {
        myPainter.drawImage(p, NaoTopImg);
        myPainter.drawImage(p2, NaoBottomImg);
    }
}

void WoZInterface::centerGaze() {
    nao_msgs::JointAnglesWithSpeed head_angle;
    head_angle.joint_names.push_back("HeadPitch");
    head_angle.joint_angles.push_back(0.1);
    head_angle.speed = 0.25;
    pub_move.publish(head_angle);
}

void WoZInterface::publishActionLabel(std::string action) {
    actionLabel.timestamp = getTimeStamp();
    actionLabel.action = action;
    pub_label_msgs.publish(actionLabel);
}

/* Gets the current timestamp whenever it is called */
std::string WoZInterface::getTimeStamp() {
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, 80, "%Y-%m-%d-%H-%M-%S", timeinfo);
    std::string str(buffer);
    return str;
}

/* Activates an action if it is called via rostopic */
void WoZInterface::actionCallback(const std_msgs::Int8& msg) {
    int act = msg.data;
    if (act == 0) {
        on_discStimulus_clicked();
    } else if (act == 1) {
        on_prompt_clicked();
        if (ui->promptLimit->intValue() > 1) {
            ros::Duration waiter(3);
            waiter.sleep();
            on_discStimulus_clicked();
        }
    } else if (act == 2) {
        on_reward_clicked();
    } else if (act == 3) {
        on_failure_clicked();
    }
}

/* Makes the nao stand, look at the participant*/
void WoZInterface::on_start_clicked() {
//    // stiffen nao and disable autonomous life
//    naoqi_bridge_msgs::BodyPoseActionGoal pose;
//    std_srvs::Empty stiff;
//    life_disable.call(stiff);
//    life_on = false;
//    client_stiff.call(stiff);
//
//    // make nao stand and look at participant
//    pose.goal.pose_name = "Stand";
//    pub_pose.publish(pose);
//    loopRate(40);
//    centerGaze();
//
//    // publish state data
//    controlstate.startRecord = true;
//    ros::Duration(0.9).sleep();
//    controlstate.timestamp = getTimeStamp();
//    pub_woz_msgs.publish(controlstate);
      std_msgs::Int8 msg = std_msgs::Int8();
      pub_start_session.publish(msg);
}

void WoZInterface::on_toggleLife_clicked() {
    // toggles whether autonomous life is enabled or disabled
    std_srvs::Empty stiff;
    if (life_on) {
        life_disable.call(stiff);
    } else {
        life_enable.call(stiff);
    }
    life_on = !life_on;
}

void WoZInterface::on_stand_clicked() {
    //Tells the robot to stand
    std_srvs::Empty stiff;
    client_wakeup.call(stiff);
}

void WoZInterface::on_rest_clicked() {
    //Tells the robot to rest
    std_srvs::Empty stiff;
    client_rest.call(stiff);
}

void WoZInterface::on_angleHead_clicked() {
    centerGaze();
}

void WoZInterface::on_startRecording_clicked() {
    //Tells the robot to begin recording
    std_srvs::Empty stiff;
    client_record_start.call(stiff);
    recording = true;
    publishActionLabel("start");
}

void WoZInterface::on_stopRecording_clicked() {
    //Tells the robot to stop recording
    std_srvs::Empty stop;
    if (recording) {
        publishActionLabel("stop");
        client_record_stop.call(stop);
    }
    recording = false;
}

/* Nao delivers a discriminative stimulus while pointing at the object */
void WoZInterface::on_discStimulus_clicked() {
    std_msgs::String words;
    ui->promptLimit->display(ui->promptLimit->intValue() - 1);
    ui->countDown->display(timeout);
    name = ui->subjectName->toPlainText().toStdString();
    words.data = "\\RSPD=70\\" + name + ", what is this?";
    controlstate.startPointing = true;
    pub_speak.publish(words);
    loopRate(15);
    pub_woz_msgs.publish(controlstate);
    publishActionLabel("sd");
}

/* Nao delivers a prompt while pointing at the object */
void WoZInterface::on_prompt_clicked() {
    int pause = 0;
    std_msgs::String words;
    std::string obj;
    //ui->promptLimit->display(ui->promptLimit->intValue() - 1);
    ui->countDown->display(shortTimeout);
    obj = ui->objectName->toPlainText().toStdString();
    switch (ui->promptLimit->intValue()) {
        case 4:
            words.data = "\\RSPD=70\\ " + obj;
            break;
        case 3:
            words.data = "\\RSPD=70\\ This is " + obj;
            break;
        case 2:
            pause = 15;
            words.data = "\\RSPD=70\\ "  + name + ", this is " + obj;
            ui->countDown->display(shortTimeout + 1);
            break;
        case 1:
            pause = 25;
            words.data = "\\RSPD=70\\ "  + name + ", say \\pau=200\\ this is " + obj;
            ui->countDown->display(8);
            break;
    }
    controlstate.startPointing2 = true;
    pub_speak.publish(words);
    loopRate(pause);
    pub_woz_msgs.publish(controlstate);
    publishActionLabel("prompt");
}

/* Nao delivers a correcting prompt, while pointing at the object  */
/*void WoZInterface::on_correctingPrompt_clicked() {
    std_msgs::String words;
    ui->promptLimit->display(ui->promptLimit->intValue() - 1);
    ui->countDown->display(timeout);
    name = ui->subjectName->toPlainText().toStdString();
    std::string obj = ui->objectName->toPlainText().toStdString();
    words.data = "\\RSPD=70\\This is " + obj + ". Can you repeat that to me " + name + "?";
    controlstate.startPointing3 = true;
    pub_speak.publish(words);
    pub_woz_msgs.publish(controlstate);
}*/

/* Rewards patient  */
void WoZInterface::on_reward_clicked() {
    std_msgs::String words;
    ui->promptLimit->display(promptLimit);
    ui->countDown->display(0);
    name = ui->subjectName->toPlainText().toStdString();
    std::string obj = ui->objectName->toPlainText().toStdString();
    words.data = "\\RSPD=70\\You are right " + name + "! Great job!";
    controlstate.startPointing3 = true;
    pub_speak.publish(words);
    pub_woz_msgs.publish(controlstate);
    publishActionLabel("reward");
}

/* Ends a failed session */
void WoZInterface::on_failure_clicked() {
    std_msgs::String words;
    ui->promptLimit->display(promptLimit);
    ui->countDown->display(0);
    name = ui->subjectName->toPlainText().toStdString();
    std::string obj = ui->objectName->toPlainText().toStdString();
    words.data = "\\RSPD=70\\This is " + obj + ". We will try again next time!";
    pub_speak.publish(words);
    publishActionLabel("failure");
}

/* Shuts down ROS and program */
void WoZInterface::on_shutDown_clicked() {
    std_srvs::Empty stop;
    if (recording)
        client_record_stop.call(stop);

    // publish shutdown to controlstate to get other nodes to terminate
    controlstate.shutdown = true;
    pub_woz_msgs.publish(controlstate);

    ros::shutdown();
    exit(0);
}

/* Updates image data and gui */
void WoZInterface::timerEvent(QTimerEvent*) {
    UpdateImage();
    update();
}

/* Call back to store image data from camera using ROS and converts it to QImage */
void WoZInterface::topImageCallback(const sensor_msgs::ImageConstPtr& msg) {
    QImage image(&(msg->data[0]), msg->width, msg->height, QImage::Format_RGB888);
    NaoTopImg = image.rgbSwapped();
    count++;
}

/* Call back to store image data from camera using ROS and converts it to QImage */
void WoZInterface::bottomImageCallback(const sensor_msgs::ImageConstPtr& msg) {
    QImage image(&(msg->data[0]), msg->width, msg->height, QImage::Format_RGB888);
    NaoBottomImg = image.rgbSwapped();
}

/* Loop rate to make NAO wait for i amount of seconds */
void WoZInterface::loopRate(int loop_times) {
    ros::Rate loop_rate(15);
    for (int i = 0; i < loop_times; i++) {
        loop_rate.sleep();
    }
}

/* WoZ Msg Callback */
void WoZInterface::controlCallback(const woz_msgs::control_states States) {
    controlstate = States;
    if (!controlstate.startPointing && !controlstate.startPointing2
        && !controlstate.startPointing3) {
        //ros::Duration(1).sleep();
        centerGaze();
    }
}