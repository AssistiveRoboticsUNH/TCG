#include "../include/sawyer_woz/sawyer_woz.hpp"
#include <QApplication>
#include <QtGui>

int main(int argc, char ** argv){
	ros::init(argc, argv, "sawyer_woz");
	QApplication a(argc, argv);
	SawyerWoZInterface w;
	w.show();
	
	return a.exec();
}
