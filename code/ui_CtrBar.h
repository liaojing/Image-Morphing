/********************************************************************************
** Form generated from reading UI file 'CtrBar.ui'
**
** Created: Sun Jun 8 13:29:04 2014
**      by: Qt User Interface Compiler version 4.8.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CTRBAR_H
#define UI_CTRBAR_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QPushButton>
#include <QtGui/QSlider>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CCtrBar
{
public:
    QSlider *Slider_range;
    QWidget *layoutWidget;
    QHBoxLayout *horizontalLayout;
    QPushButton *pushButton_record;
    QPushButton *pushButton_play;
    QPushButton *pushButton_pause;
    QPushButton *pushButton_stop;

    void setupUi(QWidget *CCtrBar)
    {
        if (CCtrBar->objectName().isEmpty())
            CCtrBar->setObjectName(QString::fromUtf8("CCtrBar"));
        CCtrBar->resize(539, 45);
        Slider_range = new QSlider(CCtrBar);
        Slider_range->setObjectName(QString::fromUtf8("Slider_range"));
        Slider_range->setGeometry(QRect(230, 10, 291, 31));
        Slider_range->setOrientation(Qt::Horizontal);
        layoutWidget = new QWidget(CCtrBar);
        layoutWidget->setObjectName(QString::fromUtf8("layoutWidget"));
        layoutWidget->setGeometry(QRect(10, 0, 196, 42));
        horizontalLayout = new QHBoxLayout(layoutWidget);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        pushButton_record = new QPushButton(layoutWidget);
        pushButton_record->setObjectName(QString::fromUtf8("pushButton_record"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icon/icon/record.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_record->setIcon(icon);
        pushButton_record->setIconSize(QSize(32, 32));

        horizontalLayout->addWidget(pushButton_record);

        pushButton_play = new QPushButton(layoutWidget);
        pushButton_play->setObjectName(QString::fromUtf8("pushButton_play"));
        QIcon icon1;
        icon1.addFile(QString::fromUtf8(":/icon/icon/play.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_play->setIcon(icon1);
        pushButton_play->setIconSize(QSize(32, 32));

        horizontalLayout->addWidget(pushButton_play);

        pushButton_pause = new QPushButton(layoutWidget);
        pushButton_pause->setObjectName(QString::fromUtf8("pushButton_pause"));
        QIcon icon2;
        icon2.addFile(QString::fromUtf8(":/icon/icon/pause.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_pause->setIcon(icon2);
        pushButton_pause->setIconSize(QSize(32, 32));

        horizontalLayout->addWidget(pushButton_pause);

        pushButton_stop = new QPushButton(layoutWidget);
        pushButton_stop->setObjectName(QString::fromUtf8("pushButton_stop"));
        QIcon icon3;
        icon3.addFile(QString::fromUtf8(":/icon/icon/stop.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_stop->setIcon(icon3);
        pushButton_stop->setIconSize(QSize(32, 32));

        horizontalLayout->addWidget(pushButton_stop);


        retranslateUi(CCtrBar);

        QMetaObject::connectSlotsByName(CCtrBar);
    } // setupUi

    void retranslateUi(QWidget *CCtrBar)
    {
        CCtrBar->setWindowTitle(QString());
        pushButton_record->setText(QString());
        pushButton_play->setText(QString());
        pushButton_pause->setText(QString());
        pushButton_stop->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class CCtrBar: public Ui_CCtrBar {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CTRBAR_H
