#include <QTextStream>
#include <QFileDialog>
#include <QDebug>
#include <QColor>
#include <QColorDialog>
#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    currentColor(QColor("#000")),
    NeurNet(new GameWidget_3(this))
{
    ui->setupUi(this);                              //Fill in the drop-down list with the types of evolution

    QPixmap icon(16, 16);
    icon.fill(currentColor);

    nEpoch = 1;
    nMode = 0;
    fRunEventsA.resize(size_t(nEpoch));
    fGoodEventsA.resize(size_t(nEpoch));
    fRunEventsT.resize(size_t(nEpoch));
    fGoodEventsT.resize(size_t(nEpoch));
    fRunEpochsL.resize(size_t(nEpoch));

    nTraining_ev = ui->trainNumber->value();
    nTest_ev = ui->testNumber->value();
    fRunEventsL.resize(size_t(nTraining_ev));
    fRunEpochsL.resize(size_t(nEpoch));

    // User interface setup for Neural Net
    ui->gameLayout_3->addWidget(NeurNet);

    ui->comboBox->addItem("FC(2,Softmax)");
    ui->comboBox->addItem("FC(64,bn,LReLU,dropout, 0.5)");
    ui->comboBox->addItem("2 FC(64,bn,LReLU,dropout, 0.5)");
    connect(ui->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(setMode(int)));

    ui->comboBox_3->addItem("Training/Test");
    ui->comboBox_3->addItem("Phi");
    ui->comboBox_3->addItem("Teta");
    ui->comboBox_3->addItem("Momentum");
    connect(ui->comboBox_3, SIGNAL(currentIndexChanged(int)), this, SLOT(setDrawType(int)));

    connect(ui->test, SIGNAL(clicked()), this, SLOT(analyzeQGP()));  // Start DIRECT MODE with press the button "Direct Mode"
    connect(ui->trainNumber, SIGNAL(valueChanged(int)), this, SLOT(setTrain(int)));     // choose TRAIN MODE number
    connect(ui->testNumber, SIGNAL(valueChanged(int)), this, SLOT(setTest(int)));     // choose TEST MODE number
    connect(ui->spinBox_3, SIGNAL(valueChanged(int)), this, SLOT(setEpoch(int)));     // choose EPOCHS number
    connect(ui->spinBox_4, SIGNAL(valueChanged(int)), this, SLOT(setBatch(int)));


    connect(ui->loadQGP, SIGNAL(clicked()), this, SLOT(loadQGP()));   //Load QGP and NQGP Data

//    connect(NeurNet,SIGNAL(iter(int)),ui->progressBar,SLOT(setValue(int)));  // show
//    connect(NeurNet, SIGNAL(hide()), ui->progressBar,SLOT(hide())); // hide Progress Bar in the end of training
//    ui->progressBar->hide();

    ui->widget_h->clearGraphs();
    ui->widget_h->addGraph();
    ui->widget_h->xAxis->setLabel("NEpoch");
    ui->widget_h->yAxis->setLabel("Efficiency");
    ui->widget_h->xAxis->setRange(0, nEpoch+1);
    ui->widget_h->yAxis->setRange(0, 110);
    ui->widget_h->replot();


    ui->widget_l->clearGraphs();
    ui->widget_l->addGraph();
    ui->widget_l->xAxis->setLabel("NEpoch");
    ui->widget_l->yAxis->setLabel("Loss");
    ui->widget_l->xAxis->setRange(0, nEpoch+1);
    ui->widget_l->yAxis->setRange(0, 60);
    ui->widget_l->replot();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setDrawType(int number)
{
    drawType = number;
    qDebug()<<" --- setDrawType: "<<drawType;
    switch(drawType) {
      case 0:
        qDebug()<<" --- drawHistoStep";
        drawHistoStep(nEpoch-1);
        break;
      case 1:
        drawPhi();
        break;
      case 2:
        drawTeta();
        break;
      case 3:
        drawMomentum();
        break;
      default:
        break;
    }
}

void MainWindow::setTrain(int number) // choose Test number
{
    nTraining_ev = number;
}

void MainWindow::setTest(int number) // choose Test number
{
    nTest_ev = number;
}

void MainWindow::setMode(int number)
{
    nMode = number;
    NeurNet->setTopology(nMode);
}

void MainWindow::setEpoch(int number) // choose Test number
{
    nEpoch = number;
    NeurNet->setEpoch(nEpoch);
    fRunEventsA.resize(size_t(nEpoch));
    fGoodEventsA.resize(size_t(nEpoch));
    fRunEventsT.resize(size_t(nEpoch));
    fGoodEventsT.resize(size_t(nEpoch));
    fRunEventsL.resize(size_t(nTraining_ev));
    fRunEpochsL.resize(size_t(nEpoch));

    ui->widget_h->xAxis->setRange(0, nEpoch+1);
    ui->widget_h->replot();

    ui->widget_l->xAxis->setRange(0, nEpoch+1);
    ui->widget_l->replot();
}

void MainWindow::setBatch(int number)
{
    NeurNet->setBatch(number);
}


void MainWindow::loadQGP() //load QGP information
{
    fDirname = QFileDialog::getExistingDirectory(this, tr("Open input data directory"), QDir::homePath());
    qDebug() << " - loadQGP - fDirname: " << fDirname;
}

void MainWindow::analysisStep(int iEp)
{
    int batch_index = 0;
    for( int iqgp = 0; iqgp < 2; iqgp++ ) {
      bool bqgp = false;
      if( iqgp == 0 ) bqgp = true;
      QString s_qgp[2] = { "/qgp/", "/nqgp/" };
      QString s_fname1 = "phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr.";
      QString s_fname2 = "_event.dat";
      for( int iEv = 0; iEv < nTraining_ev; iEv++ ) {
        QString filename = fDirname + s_qgp[iqgp] + s_fname1 + QString::number(iEv) + s_fname2;
        std::string filename_str = filename.toUtf8().constData();
        qDebug() << filename;
        bool good = NeurNet->AnalyzeMode(filename_str,bqgp,batch_index);
        batch_index++;
        fRunEventsA[size_t(iEp)]++;
        fRunEventsL[size_t(iEv)] = NeurNet->getLoss();
        if( good ) fGoodEventsA[size_t(iEp)]++;
      }
      double summL = 0;
      for(int iEv = 0; iEv < nTraining_ev; iEv++ )
      {
          summL += fRunEventsL[size_t(iEv)];
      }
      fRunEpochsL[iEp] = summL/nTraining_ev;
    }
//    NeurNet->net_ending();
}

void MainWindow::analysisStep1(int iEp)
{
      bool bqgp = true;
      QString s_qgp[2] = { "/qgp/", "/nqgp/" };
      QString s_fname1 = "phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr.";
      QString s_fname2 = "_event.dat";
      int batch_index = 0;
      for( int iEv = 0; iEv < nTraining_ev; iEv++ ) {
          for( int iqgp = 0; iqgp < 2; iqgp++ ) {
              if( iqgp != 0 ) bqgp = false;
              else bqgp = true;
              QString filename = fDirname + s_qgp[iqgp] + s_fname1 + QString::number(iEv) + s_fname2;
              std::string filename_str = filename.toUtf8().constData();
              qDebug() << filename;
//              bool good = NeurNet->AnalyzeMode(filename_str,bqgp,batch_index);

              int good = 0;
              NeurNet->LoadBatch(filename_str,bqgp,batch_index);
              batch_index++;
              if(batch_index == NeurNet->getBatchSize()){
                  good = NeurNet->AnalyzeModeBatch();
                  batch_index = 0;
              }
              fRunEventsA[size_t(iEp)]++;
              fRunEventsL[size_t(iEp)] = NeurNet->getLoss();
//              if( good ) fGoodEventsA[size_t(iEp)]++;
              fGoodEventsA[size_t(iEp)]+=good;

          }
          double summL = 0;
          for(int iEv = 0; iEv < nTraining_ev; iEv++ )
          {
              summL += fRunEventsL[size_t(iEv)];
          }
          fRunEpochsL[iEp] = summL/nTraining_ev;
      }
      NeurNet->PrintStat();
//    NeurNet->net_ending();
}

void MainWindow::testStep(int iEp)
{
    for( int iqgp = 0; iqgp < 2; iqgp++ ) {
        bool bqgp = true;
        if( iqgp != 0 ) bqgp = false;
        else bqgp = true;
        QString s_qgp[2] = { "/qgp/", "/nqgp/" };
        QString s_fname1 = "phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr.";
        QString s_fname2 = "_event.dat";
        for( int iEv = nTraining_ev; iEv < nTraining_ev+nTest_ev; iEv++ ) {
//        for( int iEv = 0; iEv < nTest_ev; iEv++ ) {
          QString filename = fDirname + s_qgp[iqgp] + s_fname1 + QString::number(iEv) + s_fname2;
          std::string filename_str = filename.toUtf8().constData();
          qDebug() << filename;
          bool good = NeurNet->TestMode(filename_str,bqgp);
          fRunEventsT[size_t(iEp)]++;
          if( good ) fGoodEventsT[size_t(iEp)]++;
        }
    }
}

void MainWindow::drawLossStep(int iEp)
{
    QVector<double> x(iEp+2), yL(iEp+2);
    for( int i = 0; i < iEp+2; i++ ) x[i] = i;
    yL[0] = 0;

    for( int i = 0; i < iEp+2; i++ ) {
        yL[i] = fRunEpochsL[size_t(i-1)];
        qDebug() << "Loss epoch " <<  yL[i];
    }

    ui->widget_l->addGraph();
    ui->widget_l->graph(0)->setData(x, yL);

    QPen pen;
    pen.setColor(QColor(25,135,25));
    pen.setWidth(2);
    ui->widget_l->graph(0)->setPen(pen);
    ui->widget_l->replot();

    double max = 0;
    for(size_t i=0; i < fRunEpochsL.size(); i++ )
    {
        if(fRunEpochsL[i] > max)
            max = fRunEpochsL[i];
    }
    max = max*1.1;

    ui->widget_l->yAxis->setRange(0, max);
    ui->widget_l->replot();
}


void MainWindow::drawHistoStep(int iEp)
{
    ui->widget_h->clearGraphs();
    ui->widget_h->addGraph();
    ui->widget_h->xAxis->setLabel("NEpoch");
    ui->widget_h->yAxis->setLabel("Efficiency");
    ui->widget_h->xAxis->setRange(0, nEpoch+1);
    ui->widget_h->yAxis->setRange(0, 110);
    ui->widget_h->replot();

    QVector<double> x(iEp+2), yA(iEp+2), yT(iEp+2);
    for( int i = 0; i < iEp+2; i++ ) x[i] = i;
    yA[0] = 0;
    yT[0] = 0;
    for( int i = 1; i < iEp+2; i++ ) {
        if(fRunEventsA[size_t(i-1)]) yA[i] = 100*fGoodEventsA[size_t(i-1)]/fRunEventsA[size_t(i-1)];
        if(fRunEventsT[size_t(i-1)]) {
          if(nTest_ev) yT[i] = 100*fGoodEventsT[size_t(i-1)]/fRunEventsT[size_t(i-1)];
          else yT[i] = 0;
        }
    }
    ui->widget_h->addGraph();
    ui->widget_h->addGraph();
    ui->widget_h->graph(0)->setData(x, yA);
    ui->widget_h->graph(1)->setData(x, yT);

    QPen pen;
    pen.setColor(QColor(25,135,25));
    pen.setWidth(2);
    ui->widget_h->graph(0)->setPen(pen);
    pen.setWidth(3);
    pen.setStyle(Qt::DashLine);
    pen.setColor(QColor(20,105,20));
    ui->widget_h->graph(1)->setPen(pen);

    ui->widget_h->replot();
}

void MainWindow::drawPhi()
{
    ui->widget_h->clearGraphs();
    ui->widget_h->addGraph();
    ui->widget_h->xAxis->setLabel("Bin");
    ui->widget_h->yAxis->setLabel("Amount");
    ui->widget_h->xAxis->setRange(0, 21);
    ui->widget_h->replot();

    int *phi_stat = NeurNet->getPhiStat();
    int *phi_stat_n = NeurNet->getPhiStat_n();
    QVector<double> x(20), y(20), y_n(20);
    float y_max = 0;
    for( int i = 0; i < 20; i++ ) x[i] = i;
    for( int i = 0; i < 20; i++ ) {
        y[i] = phi_stat[i];
        y_n[i] = phi_stat_n[i];
        if( y[i] > y_max ) y_max = y[i];
        if( y_n[i] > y_max ) y_max = y_n[i];
    }
    ui->widget_h->yAxis->setRange(0, y_max+y_max*0.1);
    ui->widget_h->addGraph();
    ui->widget_h->addGraph();
    ui->widget_h->graph(0)->setData(x, y);
    ui->widget_h->graph(1)->setData(x, y_n);
    QPen pen;
    pen.setColor(QColor(25,135,25));
    pen.setWidth(2);
    ui->widget_h->graph(0)->setPen(pen);
    pen.setColor(QColor(135,25,25));
    pen.setStyle(Qt::DashLine);
    pen.setWidth(2);
    ui->widget_h->graph(1)->setPen(pen);
    ui->widget_h->replot();
}

void MainWindow::drawTeta()
{
    ui->widget_h->clearGraphs();
    ui->widget_h->addGraph();
    ui->widget_h->xAxis->setLabel("Bin");
    ui->widget_h->yAxis->setLabel("Amount");
    ui->widget_h->xAxis->setRange(0, 21);
    ui->widget_h->replot();

    int *teta_stat = NeurNet->getTetStat();
    int *teta_stat_n = NeurNet->getTetStat_n();
    QVector<double> x(20), y(20), y_n(20);
    float y_max = 0;
    for( int i = 0; i < 20; i++ ) x[i] = i;
    for( int i = 0; i < 20; i++ ) {
        y[i] = teta_stat[i];
        y_n[i] = teta_stat_n[i];
        if( y[i] > y_max ) y_max = y[i];
        if( y_n[i] > y_max ) y_max = y_n[i];
    }
    ui->widget_h->yAxis->setRange(0, y_max+y_max*0.1);
    ui->widget_h->addGraph();
    ui->widget_h->addGraph();
    ui->widget_h->graph(0)->setData(x, y);
    ui->widget_h->graph(1)->setData(x, y_n);
    QPen pen;
    pen.setColor(QColor(25,135,25));
    pen.setWidth(2);
    ui->widget_h->graph(0)->setPen(pen);
    pen.setColor(QColor(135,25,25));
    pen.setStyle(Qt::DashLine);
    pen.setWidth(2);
    ui->widget_h->graph(1)->setPen(pen);
    ui->widget_h->replot();
}

void MainWindow::drawMomentum()
{
    ui->widget_h->clearGraphs();
    ui->widget_h->addGraph();
    ui->widget_h->xAxis->setLabel("Bin (log scale)");
    ui->widget_h->yAxis->setLabel("Amount");
    ui->widget_h->xAxis->setRange(0, 21);
    ui->widget_h->replot();

    int *momentum_stat = NeurNet->getMomStat();
    int *momentum_stat_n = NeurNet->getMomStat_n();
    QVector<double> x(20), y(20), y_n(20);
    float y_max = 0;
    for( int i = 0; i < 20; i++ ) x[i] = i;
    for( int i = 0; i < 20; i++ ) {
        y[i] = momentum_stat[i];
        y_n[i] = momentum_stat_n[i];
        if( y[i] > y_max ) y_max = y[i];
        if( y_n[i] > y_max ) y_max = y_n[i];
    }
    ui->widget_h->yAxis->setRange(0, y_max+y_max*0.1);
    ui->widget_h->addGraph();
    ui->widget_h->addGraph();
    ui->widget_h->graph(0)->setData(x, y);
    ui->widget_h->graph(1)->setData(x, y_n);
    QPen pen;
    pen.setColor(QColor(25,135,25));
    pen.setWidth(2);
    ui->widget_h->graph(0)->setPen(pen);
    pen.setColor(QColor(135,25,25));
    pen.setStyle(Qt::DashLine);
    pen.setWidth(2);
    ui->widget_h->graph(1)->setPen(pen);
    ui->widget_h->replot();
}

void MainWindow::analyzeQGP()
{
    fRunEventsA.resize(size_t(nEpoch));
    fGoodEventsA.resize(size_t(nEpoch));
    fRunEventsT.resize(size_t(nEpoch));
    fGoodEventsT.resize(size_t(nEpoch));
    fRunEventsL.resize(size_t(nTraining_ev));
    fRunEpochsL.resize(size_t(nEpoch));
    for( int i = 0; i < size_t(nEpoch); i++ ) {
        fRunEventsA[i] = 0;
        fGoodEventsA[i] = 0;
        fRunEventsT[i] = 0;
        fGoodEventsT[i] = 0;
        fRunEpochsL[i] = 0;
    }
    for( int i = 0; i < size_t(nTraining_ev); i++ ) {
        fRunEventsL[i] = 0;
    }
    for( int iE = 0; iE < nEpoch; iE++ ) {
      NeurNet->clear_stat();
//      analysisStep(iE);
      analysisStep1(iE);
      if(nTest_ev) testStep(iE);
      drawHistoStep(iE);
      drawLossStep(iE);
      qDebug() << "\n\n\n> fRunEventsA: " << fRunEventsA[iE] << ";   fGoodEventsA: " << fGoodEventsA[iE] << "\n> fRunEventsT: " << fRunEventsT[iE] << ";   fGoodEventsT: " << fGoodEventsT[iE];
      QTime dieTime= QTime::currentTime().addMSecs(10);
      while (QTime::currentTime() < dieTime)
          QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
    }
    NeurNet->net_ending();
}
