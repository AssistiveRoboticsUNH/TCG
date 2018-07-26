#include "../include/woz_interface/woz_interface.hpp"
#include <QApplication>
#include <QtGui>

int main(int argc, char ** argv){
	ros::init(argc, argv, "woz_interface");
	QApplication a(argc, argv);
	WoZInterface w;
	w.show();
	
	return a.exec();
}
