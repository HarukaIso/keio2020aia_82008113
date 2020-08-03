##
from sonar.sonar_data import *
from sonar.sonar_model import *
from sonar.sonar_trainer import *
from cats.cat_data import *
from cats.cat_model import *
from cats.cat_trainer import *
##

##
def main():

    data = Sonar_Data('/Users/HARU/lecture/data/') #class sonar_dataをobject dataに格納　引数はfileのあるフォルダ
    #この操作をクラスをインスタンス化するという
    model = Sonar_Model() #class sonar_modelをmodelに格納　引数なし
    trainer = Sonar_Trainer(data, model) #sonar_trainerをtrainerに入れる　引数は上のdataとmodel
    costs, accuracies = trainer.train(0.05, 200) #sonar_trainerの中のtrain関数を用いる 引数はlrハイパーパラメター,neはnumber of epoch
    model.save_model()

    #data = Cat_Data('/Users/HARU/lecture/data/')
    #model = Cat_Model()
    #trainer = Cat_Trainer(data, model)
    #costs, accuracies = trainer.train(lr, ne)
    #model.save_model()
##

##
if __name__ == "__main__":
    main()
##
