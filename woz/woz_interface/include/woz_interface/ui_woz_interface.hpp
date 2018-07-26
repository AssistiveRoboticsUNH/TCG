#ifndef UI_WOZ_INTERFACE_H
#define UI_WOZ_INTERFACE_H

#include <QVariant>
#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QHeaderView>
#include <QLCDNumber>
#include <QPushButton>
#include <QWidget>
#include <QPalette>
#include <QTextEdit>

QT_BEGIN_NAMESPACE

class Ui_WoZInterface {
public:
    QPushButton *discStimulus;
    QPushButton *prompt;
    //QPushButton *correctingPrompt;
    QPushButton *reward;
    QPushButton *failure;
    QPushButton *stand;
    QPushButton *rest;
    QPushButton *startRecording;
    QPushButton *stopRecording;
    QPushButton *angleHead;
    QPushButton *toggleLife;
    QPushButton *shutDown;
    QPushButton *start;

    QTextEdit *subjectName;
    QTextEdit *objectName;
    //QTextEdit *decoyName;

    QLabel *subjectNameLabel;
    QLabel *objectNameLabel;
    //QLabel *decoyNameLabel;

    QLCDNumber *clock;
    QLCDNumber *countDown;
    QLCDNumber *promptLimit;

    QLabel *clockLabel;
    QLabel *countDownLabel;
    QLabel *promptLimitLabel;

    void setupUi(QWidget *WoZInterface) {
        if (WoZInterface->objectName().isEmpty())
            WoZInterface->setObjectName(QString("WoZInterface"));
        int blockw = 173;
        int blockh = 40;
        int lcdh = 80;
        int buffer = 20;
        int windowW = 600;
        int windowH = 840;

        WoZInterface->resize(windowW, windowH);

        discStimulus = new QPushButton(WoZInterface);
        discStimulus->setObjectName(QString("discStimulus"));
        discStimulus->setGeometry(QRect(20, 20, blockw, blockh));

        prompt = new QPushButton(WoZInterface);
        prompt->setObjectName(QString("prompt"));
        prompt->setGeometry(QRect(buffer, blockh + buffer * 2, blockw, blockh));

        /*correctingPrompt = new QPushButton(WoZInterface);
        correctingPrompt->setObjectName(QString("correctingPrompt"));
        correctingPrompt->setGeometry(QRect(buffer, blockh * 2 + buffer * 3,
                                            blockw, blockh));*/

        reward = new QPushButton(WoZInterface);
        reward->setObjectName(QString("reward"));
        reward->setGeometry(QRect(blockw + buffer * 2, buffer, blockw, blockh));

        failure = new QPushButton(WoZInterface);
        failure->setObjectName(QString("failure"));
        failure->setGeometry(QRect(blockw + buffer * 2, blockh + buffer * 2,
                                   blockw, blockh));

        stand = new QPushButton(WoZInterface);
        stand->setObjectName(QString("stand"));
        stand->setGeometry(QRect(blockw + buffer * 2, blockh * 2 + buffer * 3,
                                 blockw, blockh));

        rest = new QPushButton(WoZInterface);
        rest->setObjectName(QString("rest"));
        rest->setGeometry(QRect(blockw + buffer * 2, blockh * 3 + buffer * 4,
                                blockw, blockh));

        startRecording = new QPushButton(WoZInterface);
        startRecording->setObjectName(QString("startRecording"));
        startRecording->setGeometry(QRect(blockw * 2 + buffer * 3, buffer,
                                          blockw, blockh));

        stopRecording = new QPushButton(WoZInterface);
        stopRecording->setObjectName(QString("stopRecording"));
        stopRecording->setGeometry(QRect(blockw * 2 + buffer * 3, blockh + buffer * 2,
                                         blockw, blockh));

        angleHead = new QPushButton(WoZInterface);
        angleHead->setObjectName(QString("angleHead"));
        angleHead->setGeometry(QRect(blockw * 2 + buffer * 3, blockh * 2 + buffer * 3,
                                     blockw, blockh));

        toggleLife = new QPushButton(WoZInterface);
        toggleLife->setObjectName(QString("toggleLife"));
        toggleLife->setGeometry(QRect(blockw * 2 + buffer * 3, blockh * 3 + buffer * 4,
                                      blockw, blockh));

        start = new QPushButton(WoZInterface);
        start->setObjectName(QString("start"));
        start->setGeometry(QRect(blockw * 2 + buffer * 3,
                                 windowH - buffer * 2 - blockh * 2,
                                 blockw, blockh));

        shutDown = new QPushButton(WoZInterface);
        shutDown->setObjectName(QString("shutDown"));
        shutDown->setGeometry(QRect(blockw * 2 + buffer * 3, windowH - blockh - buffer,
                                    blockw, blockh));

        subjectNameLabel = new QLabel(WoZInterface);
        subjectNameLabel->setObjectName(QString("subjectNameLabel"));
        subjectNameLabel->setGeometry(QRect(buffer, blockh * 4 + buffer * 5,
                                            blockw, blockh / 2));

        objectNameLabel = new QLabel(WoZInterface);
        objectNameLabel->setObjectName(QString("objectNameLabel"));
        objectNameLabel->setGeometry(QRect(blockw + buffer * 2, blockh * 4 + buffer * 5,
                                           blockw, blockh / 2));

        /*decoyNameLabel = new QLabel(WoZInterface);
        decoyNameLabel->setObjectName(QString("decoyNameLabel"));
        decoyNameLabel->setGeometry(QRect(blockw * 2 + buffer * 3, blockh * 4 + buffer * 5,
                                          blockw, blockh / 2));*/

        subjectName = new QTextEdit(WoZInterface);
        subjectName->setObjectName(QString("subjectName"));
        subjectName->setGeometry(QRect(buffer, blockh * 4 + buffer * 6, blockw, blockh));

        objectName = new QTextEdit(WoZInterface);
        objectName->setObjectName(QString("objectName"));
        objectName->setGeometry(QRect(blockw + buffer * 2, blockh * 4 + buffer * 6,
                                      blockw, blockh));

        /*decoyName = new QTextEdit(WoZInterface);
        decoyName->setObjectName(QString("decoyName"));
        decoyName->setGeometry(QRect(blockw * 2 + buffer * 3, blockh * 4 + buffer * 6,
                                     blockw, blockh));*/

        countDownLabel = new QLabel(WoZInterface);
        countDownLabel->setObjectName(QString("countDownLabel"));
        countDownLabel->setGeometry(QRect(blockw * 2 + buffer * 3, blockh * 5 + buffer * 7,
                                          blockw, blockh / 2));

        countDown = new QLCDNumber(WoZInterface);
        countDown->setObjectName(QString("countDown"));
        countDown->setGeometry(QRect(blockw * 2 + buffer * 3, blockh * 5 + buffer * 8,
                                     blockw, lcdh));

        promptLimitLabel = new QLabel(WoZInterface);
        promptLimitLabel->setObjectName(QString("promptLimitLabel"));
        promptLimitLabel->setGeometry(QRect(blockw * 2 + buffer * 3,
                                            blockh * 5 + buffer * 9 + lcdh,
                                            blockw, blockh / 2));

        promptLimit = new QLCDNumber(WoZInterface);
        promptLimit->setObjectName(QString("promptLimit"));
        promptLimit->setGeometry(QRect(blockw * 2 + buffer * 3,
                                       blockh * 5 + buffer * 10 + lcdh,
                                       blockw, lcdh));

        clockLabel = new QLabel(WoZInterface);
        clockLabel->setObjectName(QString("clockLabel"));
        clockLabel->setGeometry(QRect(blockw * 2 + buffer * 3,
                                      blockh * 5 + buffer * 11 + lcdh * 2,
                                      blockw, blockh / 2));

        clock = new QLCDNumber(WoZInterface);
        clock->setObjectName(QString("clock"));
        clock->setGeometry(QRect(blockw * 2 + buffer * 3,
                                 blockh * 5 + buffer * 12 + lcdh * 2,
                                 blockw, lcdh));

        retranslateUi(WoZInterface);

        QMetaObject::connectSlotsByName(WoZInterface);
    }

    void retranslateUi(QWidget *WoZInterface) {
        WoZInterface->setWindowTitle(QApplication::translate("WoZInterface",
                                                             "Wizard of Oz Interface", 0));

        discStimulus->setText(QApplication::translate("WoZInterface", "Deliver SD", 0));
        prompt->setText(QApplication::translate("WoZInterface", "Prompt", 0));
        /*correctingPrompt->setText(QApplication::translate("WoZInterface",
                                                          "Correcting Prompt", 0));*/

        reward->setText(QApplication::translate("WoZInterface", "Reward", 0));
        failure->setText(QApplication::translate("WoZInterface", "Failure", 0));
        rest->setText(QApplication::translate("WoZInterface", "Rest", 0));
        stand->setText(QApplication::translate("WoZInterface", "Stand", 0));

        startRecording->setText(QApplication::translate("WoZInterface", "Start Recording", 0));
        stopRecording->setText(QApplication::translate("WoZInterface", "Stop Recording", 0));
        angleHead->setText(QApplication::translate("WoZInterface", "Angle Head", 0));
        toggleLife->setText(QApplication::translate("WoZInterface", "Toggle Life", 0));

        start->setText(QApplication::translate("WoZInterface", "Start", 0));
        shutDown->setText(QApplication::translate("WoZInterface", "Shut Down", 0));

        subjectNameLabel->setText(QApplication::translate("WoZInterface", "Subject Name", 0));
        objectNameLabel->setText(QApplication::translate("WoZInterface", "Object", 0));
        //decoyNameLabel->setText(QApplication::translate("WoZInterface", "Decoy", 0));

        countDownLabel->setText(QApplication::translate("WoZInterface", "Response Countdown", 0));
        promptLimitLabel->setText(QApplication::translate("WoZInterface",
                                                          "Prompts remaining", 0));
        clockLabel->setText(QApplication::translate("WoZInterface", "Clock", 0));

        promptLimit->display(5);
    }
};

namespace Ui {
    class WoZInterface : public Ui_WoZInterface {
    };
}

QT_END_NAMESPACE

#endif
