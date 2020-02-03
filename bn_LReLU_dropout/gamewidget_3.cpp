#include <QMessageBox>
#include <QMouseEvent>
#include <QDebug>
#include <QRectF>
#include <QPainter>
#include <qmath.h>
#include "gamewidget_3.h"
#include "QTime"

#include <fstream>
#include <iostream>
#include <iomanip>      // std::setw




GameWidget_3::GameWidget_3(QWidget *parent) :
    QWidget(parent),
    fieldSize(28),
    Neuronet(topology),
    test_results (test_file.c_str()),
    topology_type(0)

{

}

GameWidget_3::~GameWidget_3()
{

}


void GameWidget_3::setEpoch(int Epoch)
{
    epoch = Epoch;
}

void GameWidget_3::setTopology(int number)
{
    topology_type  = number;

    switch (topology_type) {
    case 0:
        topology.resize(2);
        topology[0] = 28*20*20*20;
        topology[1] = 2;
        break;
    case 1:
        topology.resize(3);
        topology[0] = 28*20*20*20;
        topology[1] = 8;
        topology[2] = 2;
        break;
    case 2:
        topology.resize(4);
        topology[0] = 28*20*20*20;
        topology[1] = 8;
        topology[2] = 8;
        topology[3] = 2;
        break;
    }

    Neuronet.change_topology(topology);
//    Neuronet.print();
}

double GameWidget_3::getLoss()
{
    double Loss = Neuronet.loss();
    return Loss;
}

void GameWidget_3::setBatch(int number)
{
    Neuronet.setBatch(number);
}

void GameWidget_3::LoadBatch(std::string filename, bool qgp, int batch_index)  // training mode of neural network
{
    qDebug()<<" --- GameWidget_3::LoadBatch --- batch_index: "<<batch_index<<"\n";
    Neuronet.loadNQGPbatch(filename,batch_index,qgp);
}

int GameWidget_3::AnalyzeModeBatch()
{
    int good_event = 0;
    if(topology_type > 0){
      Neuronet.batchNormalization();
      Neuronet.setBatch_active_neurons();
    }
    for(int i = 0; i < Neuronet.getBatchSize(); i++){
      std::vector <double> target_values;
      target_values.resize(2);
      if(Neuronet.getBatch_qgp(i)){
        target_values[0] = 1;
        target_values[1] = 0;
      }
      else{
        target_values[0] = 0;
        target_values[1] = 1;
      }

      Neuronet.fillNeurons(i);

      std::vector <double> results_values;
      std::vector <double> results_values2;

//      qDebug()<<"topology_type "<<topology_type;

      if( topology_type == 0 ) {
        Neuronet.analyze();
        Neuronet.back_prop(target_values, size_t(test), epoch, topology_type);
        test++;
        Neuronet.get_results(results_values);

        Neuronet.analyze();
        Neuronet.get_results(results_values2);
      }

      if( topology_type == 1 ) {
        Neuronet.analyze_1hidden_layer();
        Neuronet.back_prop(target_values, size_t(test), epoch, topology_type);
        test++;
        Neuronet.get_results(results_values);

        Neuronet.analyze_1hidden_layer();
        Neuronet.get_results(results_values2);
      }

//      if( topology_type == 2 ) {                                                          // Трехслойная сеть
//        Neuronet.analyze_2hidden_layer();
//        Neuronet.back_prop(target_values, size_t(test), epoch, topology_type);
//        test++;
//        Neuronet.get_results(results_values);

//        Neuronet.analyze_2hidden_layer();
//        Neuronet.get_results(results_values2);
//      }

      double max_value = 0;
      size_t max_index = 0;
      for (size_t i = 0; i != results_values.size(); ++i) {
        if (results_values[i] > max_value) {
            max_value = results_values[i];
            max_index = i;
            }
      }
      qDebug() << "results_values: " << results_values[0] << "   " << results_values[1];
      qDebug() << "results_values NEW: " << results_values2[0] << "   " << results_values2[1];
      if( Neuronet.getBatch_qgp(i) && results_values[0] > results_values[1] ) {
          qDebug() << "correct";
          good_event++;
      }
      if( !Neuronet.getBatch_qgp(i) && results_values[0] < results_values[1] ) {
          qDebug() << "correct";
          good_event++;
      }

    }
    return good_event;
}


bool GameWidget_3::AnalyzeMode(std::string filename, bool qgp, int batch_index)  // training mode of neural network
{
    QFile ff( QString::fromStdString(filename) );
    if( !ff.exists() ) {
      qDebug() << "Error! File do not exists!";
      return false;
    }
//    while(test < end){
//      qDebug() << " while(test < end) test: " << test;
    qDebug() << " - qgp: " << qgp;
//      Neuronet.loadQGP(filename);
//    Neuronet.loadNQGPbatch(filename,batch_index);

      std::vector <double> target_values;
      target_values.resize(2);
      if(qgp){
        target_values[0] = 1;
        target_values[1] = 0;
      }
      else{
        target_values[0] = 0;
        target_values[1] = 1;
      }

      std::vector <double> results_values;
      std::vector <double> results_values2;

      if( topology_type == 0 ) {
        Neuronet.analyze();
        Neuronet.back_prop(target_values, size_t(test), epoch, topology_type);
        test++;
        Neuronet.get_results(results_values);

        Neuronet.analyze();
        Neuronet.get_results(results_values2);
      }

      if( topology_type == 1 ) {
        Neuronet.analyze_1hidden_layer();
        Neuronet.back_prop(target_values, size_t(test), epoch, topology_type);
        test++;
        Neuronet.get_results(results_values);

        Neuronet.analyze_1hidden_layer();
        Neuronet.get_results(results_values2);
      }

      double max_value = 0;
      size_t max_index = 0;
      for (size_t i = 0; i != results_values.size(); ++i) {
        if (results_values[i] > max_value) {
            max_value = results_values[i];
            max_index = i;
            }
      }
      qDebug() << "results_values: " << results_values[0] << "   " << results_values[1];
      qDebug() << "results_values NEW: " << results_values2[0] << "   " << results_values2[1];
      if( qgp && results_values[0] > results_values[1] ) {
          qDebug() << "correct";
          return true;
      }
      if( !qgp && results_values[0] < results_values[1] ) {
          qDebug() << "correct";
          return true;
      }

      return false;

//    }
//    net_ending();
//    test = 0;
}
bool GameWidget_3::TestMode(std::string filename, bool qgp)  // training mode of neural network
{
    QFile ff( QString::fromStdString(filename) );
    if( !ff.exists() ) {
      qDebug() << "Error! File do not exists!";
      return false;
    }
//    while(test < end){
//      qDebug() << " while(test < end) test: " << test;
      Neuronet.loadQGP(filename);
      Neuronet.analyze();                                       // Изменить функции анализа для разных типов сетей.

      std::vector <double> target_values;
      target_values.resize(2);
      if(qgp){
        target_values[0] = 1;
        target_values[1] = 0;
      }
      else{
        target_values[0] = 0;
        target_values[1] = 1;
      }

      std::vector <double> results_values;
      Neuronet.get_results(results_values);

      double max_value = 0;
      size_t max_index = 0;
      for (size_t i = 0; i != results_values.size(); ++i) {
        if (results_values[i] > max_value) {
            max_value = results_values[i];
            max_index = i;
            }
      }
      qDebug() << "results_values: " << results_values[0] << "   " << results_values[1];

      if( qgp && results_values[0] > 0.5 && results_values[1] < 0.5 ) {
          qDebug() << "correct";
          return true;
      }
      if( !qgp && results_values[0] < 0.5 && results_values[1] > 0.5 ) {
          qDebug() << "correct";
          return true;
      }

      return false;
}



void GameWidget_3::net_ending() //end of Training Mode
{

    QMessageBox::information(this,
                                 tr("end"),
                                 tr("End of training"),
                                 QMessageBox::Ok);
}

void GameWidget_3::loadQGP( std::string filename ) //load QGP Data
{
    Neuronet.loadQGP(filename);

//    QMessageBox::information(this,
//                                 tr("loading"),
//                                 tr("loading completed"),
//                                 QMessageBox::Ok);
}

void GameWidget_3::loadNQGP( std::string filename )  //load NQGP Data
{
    Neuronet.loadNQGP(filename);

//    QMessageBox::information(this,
//                                 tr("load"),
//                                 tr("loading completed"),
//                                 QMessageBox::Ok);
}


void GameWidget_3::loadNQGPbatch( std::string filename, int batch_index, bool qgp )  //load NQGP Data
{
    Neuronet.loadNQGPbatch(filename, batch_index, qgp);

//    QMessageBox::information(this,
//                                 tr("load"),
//                                 tr("loading completed"),
//                                 QMessageBox::Ok);
}

