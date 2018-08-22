#ifndef SAWYER_UI_WOZ
#define SAWYER_UI_WOZ

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

class Ui_SawyerWoZInterface {
public:
    QPushButton *pickItem;
    QPushButton *placeDQA;
    QPushButton *placeSQA;
    QPushButton *placeBox;
    QPushButton *startRecording;
    QPushButton *stopRecording;
    QPushButton *moveToStart;
    QPushButton *run;

    QLCDNumber *countDown;
    QLCDNumber *itemLimit;

    QLabel *countDownLabel;
    QLabel *itemLimitLabel;

    void setupUi(QWidget *SawyerWoZInterface) {
        if (SawyerWoZInterface->objectName().isEmpty())
            SawyerWoZInterface->setObjectName(QString("SawyerWoZInterface"));
        int blockw = 173;
        int blockh = 40;
        int lcdh = 80;
        int buffer = 20;
        int windowW = 600;
        int windowH = 350;

        SawyerWoZInterface->resize(windowW, windowH);

        pickItem = new QPushButton(SawyerWoZInterface);
        pickItem->setObjectName(QString("pickItem"));
        pickItem->setGeometry(QRect(20, 20, blockw, blockh));

        placeDQA = new QPushButton(SawyerWoZInterface);
        placeDQA->setObjectName(QString("placeDQA"));
        placeDQA->setGeometry(QRect(blockw + buffer * 2, buffer, blockw, blockh));

        placeSQA = new QPushButton(SawyerWoZInterface);
        placeSQA->setObjectName(QString("placeSQA"));
        placeSQA->setGeometry(QRect(blockw + buffer * 2, blockh + buffer * 2, blockw, blockh));

        placeBox = new QPushButton(SawyerWoZInterface);
        placeBox->setObjectName(QString("placeBox"));
        placeBox->setGeometry(QRect(blockw + buffer * 2, blockh * 2 + buffer * 3, blockw, blockh));

        startRecording = new QPushButton(SawyerWoZInterface);
        startRecording->setObjectName(QString("startRecording"));
        startRecording->setGeometry(QRect(blockw * 2 + buffer * 3, buffer, blockw, blockh));

        stopRecording = new QPushButton(SawyerWoZInterface);
        stopRecording->setObjectName(QString("stopRecording"));
        stopRecording->setGeometry(QRect(blockw * 2 + buffer * 3, blockh + buffer * 2,
                                         blockw, blockh));

        moveToStart = new QPushButton(SawyerWoZInterface);
        moveToStart->setObjectName(QString("moveToStart"));
        moveToStart->setGeometry(QRect(blockw * 2 + buffer * 3, blockh * 2 + buffer * 3,
                                       blockw, blockh));

        run = new QPushButton(SawyerWoZInterface);
        run->setObjectName(QString("run"));
        run->setGeometry(QRect(blockw * 2 + buffer * 3, blockh * 3 + buffer * 4, blockw, blockh));

        countDownLabel = new QLabel(SawyerWoZInterface);
        countDownLabel->setObjectName(QString("countDownLabel"));
        countDownLabel->setGeometry(QRect(buffer, blockh + buffer * 4  + lcdh,
                                          blockw, blockh / 2));

        countDown = new QLCDNumber(SawyerWoZInterface);
        countDown->setObjectName(QString("countDown"));
        countDown->setGeometry(QRect(buffer, blockh + buffer * 5  + lcdh,
                                     blockw, lcdh));

        itemLimitLabel = new QLabel(SawyerWoZInterface);
        itemLimitLabel->setObjectName(QString("itemLimitLabel"));
        itemLimitLabel->setGeometry(QRect(buffer, blockh + buffer * 2,
                                          blockw, blockh / 2));

        itemLimit = new QLCDNumber(SawyerWoZInterface);
        itemLimit->setObjectName(QString("itemLimit"));
        itemLimit->setGeometry(QRect(buffer, blockh + buffer * 3,
                                     blockw, lcdh));

        retranslateUi(SawyerWoZInterface);

        QMetaObject::connectSlotsByName(SawyerWoZInterface);
    }

    void retranslateUi(QWidget *SawyerWoZInterface) {
        SawyerWoZInterface->setWindowTitle(QApplication::translate("SawyerWoZInterface",
                                                             "Wizard of Oz Interface", 0));

        pickItem->setText(QApplication::translate("SawyerWoZInterface", "Pick Item", 0));

        placeDQA->setText(QApplication::translate("SawyerWoZInterface", "Place in D. QA", 0));
        placeSQA->setText(QApplication::translate("SawyerWoZInterface", "Place in S. QA", 0));
        placeBox->setText(QApplication::translate("SawyerWoZInterface", "Place in Box", 0));

        startRecording->setText(QApplication::translate("SawyerWoZInterface", "Start Recording", 0));
        stopRecording->setText(QApplication::translate("SawyerWoZInterface", "Stop Recording", 0));
        moveToStart->setText(QApplication::translate("SawyerWoZInterface", "Reset Position", 0));

        run->setText(QApplication::translate("SawyerWoZInterface", "Start", 0));

        countDownLabel->setText(QApplication::translate("SawyerWoZInterface", "Response Countdown", 0));
        itemLimitLabel->setText(QApplication::translate("SawyerWoZInterface", "Items remaining", 0));
        itemLimit->display(3);
    }
};

namespace Ui {
    class SawyerWoZInterface : public Ui_SawyerWoZInterface {
    };
}

QT_END_NAMESPACE

#endif
