package eus.ehu.tss;


import ij.IJ;
import ij.ImagePlus;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;

public class evaluation {
    public static void main(final String[] args){
        TrainableSuperpixelSegmentation tss = new TrainableSuperpixelSegmentation();

        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<>();
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Mean"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Median"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Mode"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Skewness"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Kurtosis"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("StdDev"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Max"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Min"));
        tss.setSelectedFeatures(selectedFeatures);

        IBk exampleClassifier = new IBk();
        tss.setClassifier(exampleClassifier);

        final ArrayList<String> classes = new ArrayList<String>();
        classes.add("background");
        classes.add("blue");
        classes.add("red");
        tss.setClasses(classes);

        for(int i=0;i<10;++i){

            System.out.println("Creating training data\n\ttest class: "+(i+1));

            Instances trainingData;
            if(i==0){
                ImagePlus trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-02.png" ).getFile() );
                ImagePlus supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-02.zip" ).getFile() );
                trainingData = RegionColorFeatures.calculateColorFeatures(trainingImage1,supImage1,selectedFeatures,classes);
                for(int j=3;j<10;++j){
                    trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-0"+j+".png" ).getFile() );
                    supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-0"+j+".zip" ).getFile() );
                    try {
                        trainingData = merge(trainingData,RegionColorFeatures.calculateColorFeatures(trainingImage1,supImage1,selectedFeatures,classes));
                    }catch (Exception e){
                        e.printStackTrace();
                    }
                }
                trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-10.png" ).getFile() );
                supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-10.zip" ).getFile() );
                try {
                    trainingData = merge(trainingData,RegionColorFeatures.calculateColorFeatures(trainingImage1,supImage1,selectedFeatures,classes));
                }catch (Exception e){
                    e.printStackTrace();
                }
            }else if(i==9) {
                ImagePlus trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-01.png" ).getFile() );
                ImagePlus supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-01.zip" ).getFile() );
                trainingData = RegionColorFeatures.calculateColorFeatures(trainingImage1,supImage1,selectedFeatures,classes);
                for(int j=2;j<10;++j){
                    trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-0"+j+".png" ).getFile() );
                    supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-0"+j+".zip" ).getFile() );
                    try {
                        trainingData = merge(trainingData,RegionColorFeatures.calculateColorFeatures(trainingImage1,supImage1,selectedFeatures,classes));
                    }catch (Exception e){
                        e.printStackTrace();
                    }
                }
            }else {
                ImagePlus trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-01.png" ).getFile() );
                ImagePlus supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-01.zip" ).getFile() );
                trainingData = RegionColorFeatures.calculateColorFeatures(trainingImage1,supImage1,selectedFeatures,classes);
                for(int j=2;j<10;++j){
                    if(i+1!=j) {
                        trainingImage1 = IJ.openImage(TestSuperpixelSegmentation.class.getResource("/eval/histogram-matched-TMA/TMA-0" + j + ".png").getFile());
                        supImage1 = IJ.openImage(TestSuperpixelSegmentation.class.getResource("/eval/superpixels/SLIC-0" + j + ".zip").getFile());
                        try {
                            trainingData = merge(trainingData,RegionColorFeatures.calculateColorFeatures(trainingImage1,supImage1,selectedFeatures,classes));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
                trainingImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-10.png" ).getFile() );
                supImage1 = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-10.zip" ).getFile() );
                try {
                    trainingData = merge(trainingData,RegionColorFeatures.calculateColorFeatures(trainingImage1,supImage1,selectedFeatures,classes));
                }catch (Exception e){
                    e.printStackTrace();
                }
            }

            //System.out.println(trainingData.toSummaryString());


        }

    }

    public static Instances merge(Instances data1, Instances data2)
            throws Exception
    {
        // Check where are the string attributes
        int asize = data1.numAttributes();
        boolean strings_pos[] = new boolean[asize];
        for(int i=0; i<asize; i++)
        {
            Attribute att = data1.attribute(i);
            strings_pos[i] = ((att.type() == Attribute.STRING) ||
                    (att.type() == Attribute.NOMINAL));
        }

        // Create a new dataset
        Instances dest = new Instances(data1);
        dest.setRelationName(data1.relationName() + "+" + data2.relationName());

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;
        while (source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            dest.add(instance);

            // Copy string attributes
            for(int i=0; i<asize; i++) {
                if(strings_pos[i]) {
                    dest.instance(dest.numInstances()-1)
                            .setValue(i,instance.stringValue(i));
                }
            }
        }

        return dest;
    }
}
