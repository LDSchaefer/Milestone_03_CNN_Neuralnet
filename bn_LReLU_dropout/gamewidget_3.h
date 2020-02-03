#ifndef GameWidget_3_H
#define GameWidget_3_H

#include <QWidget>
#include "Net.h"

class GameWidget_3 : public QWidget
{
    Q_OBJECT
public:
    explicit GameWidget_3(QWidget *parent);
    ~GameWidget_3();

protected:

signals:
    //when one of the cell has been changed,emit this signal to lock the universeSize
    void environmentChanged(bool ok);
    void environmentChanged2(bool ok);
    void iter(int number);
    void hide();


    //when game is over or clear is called,emit it to unlock the universeSize
    void gameEnds(bool ok);

public slots:

    bool AnalyzeMode(std::string filename, bool qgp, int batch_index);
    bool TestMode(std::string filename, bool qgp);

    void net_ending(); //end of Training Mode
    void loadQGP(std::string filename);  //load QGP Data
    void loadNQGP(std::string filename);  //load NQGP Data
    void setEpoch(int Epoch);
    void setBatch(int number);
    void setTopology(int number);

    void LoadBatch(std::string filename, bool qgp, int batch_index);
    int AnalyzeModeBatch();

    // stat
    void PrintStat() {
        Neuronet.PrintStat();
    }
    int* getPhiStat() { return &Neuronet.getPhiStat()[0]; }
    int* getTetStat() { return &Neuronet.getTetStat()[0]; }
    int* getMomStat() { return &Neuronet.getMomStat()[0]; }

    int* getPhiStat_n() { return &Neuronet.getPhiStat_n()[0]; }
    int* getTetStat_n() { return &Neuronet.getTetStat_n()[0]; }
    int* getMomStat_n() { return &Neuronet.getMomStat_n()[0]; }

    void clear_stat() { Neuronet.clear_stat(); }
    //


    int getBatchSize()
    {
        return Neuronet.getBatchSize();
    }

    void loadNQGPbatch( std::string filename, int batch_index, bool qgp );


    double getLoss();

private slots:

private:
    int fieldSize;

    int images_number, images_size = 28, start, epoch, test=0, end = 10000, mnist_number = 0;
    std::vector <std::vector <double>> data_big, data;
    std::vector <double> labels_big, labels;
    bool readI = false, readL = false;

    std::vector <size_t> topology = /*{ 784, 64, 10 }; */ {28*20*20*20, 2}; //---
    Net Neuronet;
    std::string test_file = "Results.txt";
    std::ofstream test_results;
    double correct = 0;

    int topology_type;

};

#endif // GameWidget_3_H
