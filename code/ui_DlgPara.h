/********************************************************************************
** Form generated from reading UI file 'DlgPara.ui'
**
** Created: Wed May 28 23:06:52 2014
**      by: Qt User Interface Compiler version 4.8.4
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DLGPARA_H
#define UI_DLGPARA_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QFormLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CDlgPara
{
public:
    QGroupBox *groupBox;
    QWidget *formLayoutWidget;
    QFormLayout *formLayout;
    QLabel *label;
    QLineEdit *lineEdit_SSIM;
    QLabel *label_2;
    QLineEdit *lineEdit_SSIMClamp;
    QLabel *label_3;
    QLineEdit *lineEdit_TPS;
    QLabel *label_4;
    QLineEdit *lineEdit_UI;
    QGroupBox *groupBox_2;
    QWidget *formLayoutWidget_2;
    QFormLayout *formLayout_2;
    QRadioButton *radioButton_1;
    QRadioButton *radioButton_2;
    QRadioButton *radioButton_3;
    QGroupBox *groupBox_3;
    QWidget *formLayoutWidget_3;
    QFormLayout *formLayout_3;
    QLabel *label_6;
    QLineEdit *lineEdit_Iter;
    QLabel *label_7;
    QLineEdit *lineEdit_Drop;
    QLabel *label_8;
    QLineEdit *lineEdit_EPS;
    QWidget *horizontalLayoutWidget;
    QHBoxLayout *horizontalLayout;
    QPushButton *pushButton_Confirm;
    QPushButton *pushButton_Cancel;

    void setupUi(QWidget *CDlgPara)
    {
        if (CDlgPara->objectName().isEmpty())
            CDlgPara->setObjectName(QString::fromUtf8("CDlgPara"));
        CDlgPara->resize(274, 391);
        groupBox = new QGroupBox(CDlgPara);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(10, 10, 251, 131));
        formLayoutWidget = new QWidget(groupBox);
        formLayoutWidget->setObjectName(QString::fromUtf8("formLayoutWidget"));
        formLayoutWidget->setGeometry(QRect(10, 20, 231, 100));
        formLayout = new QFormLayout(formLayoutWidget);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        formLayout->setContentsMargins(0, 0, 0, 0);
        label = new QLabel(formLayoutWidget);
        label->setObjectName(QString::fromUtf8("label"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label);

        lineEdit_SSIM = new QLineEdit(formLayoutWidget);
        lineEdit_SSIM->setObjectName(QString::fromUtf8("lineEdit_SSIM"));

        formLayout->setWidget(0, QFormLayout::FieldRole, lineEdit_SSIM);

        label_2 = new QLabel(formLayoutWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        formLayout->setWidget(1, QFormLayout::LabelRole, label_2);

        lineEdit_SSIMClamp = new QLineEdit(formLayoutWidget);
        lineEdit_SSIMClamp->setObjectName(QString::fromUtf8("lineEdit_SSIMClamp"));

        formLayout->setWidget(1, QFormLayout::FieldRole, lineEdit_SSIMClamp);

        label_3 = new QLabel(formLayoutWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        formLayout->setWidget(2, QFormLayout::LabelRole, label_3);

        lineEdit_TPS = new QLineEdit(formLayoutWidget);
        lineEdit_TPS->setObjectName(QString::fromUtf8("lineEdit_TPS"));

        formLayout->setWidget(2, QFormLayout::FieldRole, lineEdit_TPS);

        label_4 = new QLabel(formLayoutWidget);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        formLayout->setWidget(3, QFormLayout::LabelRole, label_4);

        lineEdit_UI = new QLineEdit(formLayoutWidget);
        lineEdit_UI->setObjectName(QString::fromUtf8("lineEdit_UI"));

        formLayout->setWidget(3, QFormLayout::FieldRole, lineEdit_UI);

        groupBox_2 = new QGroupBox(CDlgPara);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        groupBox_2->setGeometry(QRect(10, 150, 251, 71));
        formLayoutWidget_2 = new QWidget(groupBox_2);
        formLayoutWidget_2->setObjectName(QString::fromUtf8("formLayoutWidget_2"));
        formLayoutWidget_2->setGeometry(QRect(10, 20, 234, 42));
        formLayout_2 = new QFormLayout(formLayoutWidget_2);
        formLayout_2->setObjectName(QString::fromUtf8("formLayout_2"));
        formLayout_2->setContentsMargins(0, 0, 0, 0);
        radioButton_1 = new QRadioButton(formLayoutWidget_2);
        radioButton_1->setObjectName(QString::fromUtf8("radioButton_1"));

        formLayout_2->setWidget(0, QFormLayout::LabelRole, radioButton_1);

        radioButton_2 = new QRadioButton(formLayoutWidget_2);
        radioButton_2->setObjectName(QString::fromUtf8("radioButton_2"));

        formLayout_2->setWidget(0, QFormLayout::FieldRole, radioButton_2);

        radioButton_3 = new QRadioButton(formLayoutWidget_2);
        radioButton_3->setObjectName(QString::fromUtf8("radioButton_3"));

        formLayout_2->setWidget(1, QFormLayout::LabelRole, radioButton_3);

        groupBox_3 = new QGroupBox(CDlgPara);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        groupBox_3->setGeometry(QRect(10, 230, 251, 111));
        formLayoutWidget_3 = new QWidget(groupBox_3);
        formLayoutWidget_3->setObjectName(QString::fromUtf8("formLayoutWidget_3"));
        formLayoutWidget_3->setGeometry(QRect(10, 20, 231, 81));
        formLayout_3 = new QFormLayout(formLayoutWidget_3);
        formLayout_3->setObjectName(QString::fromUtf8("formLayout_3"));
        formLayout_3->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
        formLayout_3->setContentsMargins(0, 0, 0, 0);
        label_6 = new QLabel(formLayoutWidget_3);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        formLayout_3->setWidget(0, QFormLayout::LabelRole, label_6);

        lineEdit_Iter = new QLineEdit(formLayoutWidget_3);
        lineEdit_Iter->setObjectName(QString::fromUtf8("lineEdit_Iter"));

        formLayout_3->setWidget(0, QFormLayout::FieldRole, lineEdit_Iter);

        label_7 = new QLabel(formLayoutWidget_3);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        formLayout_3->setWidget(1, QFormLayout::LabelRole, label_7);

        lineEdit_Drop = new QLineEdit(formLayoutWidget_3);
        lineEdit_Drop->setObjectName(QString::fromUtf8("lineEdit_Drop"));

        formLayout_3->setWidget(1, QFormLayout::FieldRole, lineEdit_Drop);

        label_8 = new QLabel(formLayoutWidget_3);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        formLayout_3->setWidget(2, QFormLayout::LabelRole, label_8);

        lineEdit_EPS = new QLineEdit(formLayoutWidget_3);
        lineEdit_EPS->setObjectName(QString::fromUtf8("lineEdit_EPS"));

        formLayout_3->setWidget(2, QFormLayout::FieldRole, lineEdit_EPS);

        horizontalLayoutWidget = new QWidget(CDlgPara);
        horizontalLayoutWidget->setObjectName(QString::fromUtf8("horizontalLayoutWidget"));
        horizontalLayoutWidget->setGeometry(QRect(10, 350, 251, 31));
        horizontalLayout = new QHBoxLayout(horizontalLayoutWidget);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        pushButton_Confirm = new QPushButton(horizontalLayoutWidget);
        pushButton_Confirm->setObjectName(QString::fromUtf8("pushButton_Confirm"));

        horizontalLayout->addWidget(pushButton_Confirm);

        pushButton_Cancel = new QPushButton(horizontalLayoutWidget);
        pushButton_Cancel->setObjectName(QString::fromUtf8("pushButton_Cancel"));

        horizontalLayout->addWidget(pushButton_Cancel);


        retranslateUi(CDlgPara);

        QMetaObject::connectSlotsByName(CDlgPara);
    } // setupUi

    void retranslateUi(QWidget *CDlgPara)
    {
        CDlgPara->setWindowTitle(QApplication::translate("CDlgPara", "Parameters", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("CDlgPara", "Weights", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("CDlgPara", "SSIM", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("CDlgPara", "SSIM clamp", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("CDlgPara", "TPS", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("CDlgPara", "UI", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("CDlgPara", "Boundary", 0, QApplication::UnicodeUTF8));
        radioButton_1->setText(QApplication::translate("CDlgPara", "no lock", 0, QApplication::UnicodeUTF8));
        radioButton_2->setText(QApplication::translate("CDlgPara", "lock corners", 0, QApplication::UnicodeUTF8));
        radioButton_3->setText(QApplication::translate("CDlgPara", "lock boundaries", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("CDlgPara", "Others", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("CDlgPara", "Iteration number", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("CDlgPara", "Drop factor", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("CDlgPara", "EPS", 0, QApplication::UnicodeUTF8));
        pushButton_Confirm->setText(QApplication::translate("CDlgPara", "Confirm", 0, QApplication::UnicodeUTF8));
        pushButton_Cancel->setText(QApplication::translate("CDlgPara", "Cancel", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class CDlgPara: public Ui_CDlgPara {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DLGPARA_H
