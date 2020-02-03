#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QColor>
#include "gamewidget_3.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void loadQGP(); //load QGP information
    void analysisStep(int iEp = 0);
    void analysisStep1(int iEp = 0);
    void testStep(int iEp = 0);
    void analyzeQGP();

    void setTrain(int number);
    void setTest(int number);
    void setEpoch(int number);
    void setBatch(int number);
    void setMode(int number);

    void setDrawType(int number);

    void drawHistoStep(int iEp = 0);
    void drawLossStep(int iEp = 0);

    void drawPhi();
    void drawTeta();
    void drawMomentum();

private:
    Ui::MainWindow *ui;
    QColor currentColor;
    GameWidget_3* NeurNet;
    std::string Filename;  /// Lade qpg-Datei CNN input
//    std::string fDirname;
    QString fDirname;
    int nTraining_ev;
    int nTest_ev;
    int nEpoch;
    int nMode;
    int drawType;
    bool qgp = true;
    std::vector<int> fRunEventsA;
    std::vector<int> fGoodEventsA;
    std::vector<int> fRunEventsT;
    std::vector<int> fGoodEventsT;
    std::vector<double> fRunEventsL;
    std::vector<double> fRunEpochsL;

};

#endif // MAINWINDOW_H
