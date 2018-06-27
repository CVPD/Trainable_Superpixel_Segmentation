package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import ij.plugin.ChannelSplitter;
import ij.process.ColorSpaceConverter;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * This class will extract region features from colored images using the Lab color space
 */
public class RegionColorFeatures {

    private static Callable<Instances> getInstances(ImagePlus inputImage,
                                                   ImagePlus labelImage,
                                                   ArrayList<RegionFeatures.Feature> selectedFeatures,
                                                   ArrayList<String> classes){
        if(Thread.currentThread().isInterrupted()){
            return null;
        }
        return new Callable<Instances>() {
            @Override
            public Instances call() throws Exception {
                Instances result =  RegionFeatures.calculateUnlabeledRegionFeatures(inputImage,labelImage,selectedFeatures,classes);
                return result;
            }
        };
    }

    /**
     * Creates Instances object based on the features of each color channel after converting the image to Lab
     * @param inputImage RGB input image
     * @param labelImage Label image
     * @param selectedFeatures ArrayList with selected features from RegionFeatures.Feature
     * @param classes ArrayList of Strings with names of the classes
     * @return Dataset with the features of each color for each region from the labelImage
     */
    public static Instances calculateUnlabeledColorFeatures(ImagePlus inputImage,
                                                            ImagePlus labelImage,
                                                            ArrayList<RegionFeatures.Feature> selectedFeatures,
                                                            ArrayList<String> classes)
    {
        ColorSpaceConverter converter = new ColorSpaceConverter();
        ImagePlus lab = converter.RGBToLab(inputImage);
        ImagePlus[] channels = ChannelSplitter.split(lab);
        if(Thread.currentThread().isInterrupted()){
            return null;
        }
        ArrayList<Instances> ins = new ArrayList<Instances>();
        ExecutorService exe = Executors.newFixedThreadPool(3);
        final ArrayList<Future< Instances > > futures = new ArrayList<Future<Instances>>();
        try {
            futures.add(exe.submit(getInstances(channels[0],labelImage,selectedFeatures,classes)));
            futures.add(exe.submit(getInstances(channels[1],labelImage,selectedFeatures,classes)));
            futures.add(exe.submit(getInstances(channels[2],labelImage,selectedFeatures,classes)));
            int i=0;
            for(Future<Instances> f : futures){
                Instances res = f.get();
                ins.add(res);
            }
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
        finally {
            exe.shutdown();
        }
        Instances lIns = ins.get(0);
        Instances aIns = ins.get(1);
        Instances bIns = ins.get(2);
        if(lIns==null||aIns==null||bIns==null){
            return null;
        }else {
            for (int i = 0; i < lIns.numAttributes(); ++i) {//all channels should have the same number of attributes
                lIns.renameAttribute(i, lIns.attribute(i).name() + "-L");
                aIns.renameAttribute(i, aIns.attribute(i).name() + "-a");
                bIns.renameAttribute(i, bIns.attribute(i).name() + "-b");
            }
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            int numAttributes = lIns.numAttributes() * 3 - 3;//-3 to remove the class attributes
            for (int i = 0; i < lIns.numAttributes() - 1; ++i) {
                attributes.add(lIns.attribute(i));
            }
            for (int i = 0; i < aIns.numAttributes() - 1; ++i) {
                attributes.add(aIns.attribute(i));
            }
            for (int i = 0; i < bIns.numAttributes() - 1; ++i) {
                attributes.add(bIns.attribute(i));
            }
            attributes.add(new Attribute("Class", classes));
            Instances unlabeled = new Instances("training data", attributes, 0);
            for (int i = 0; i < lIns.numInstances(); ++i) {
                int k = 0;
                Instance inst = new DenseInstance(numAttributes + 1);
                for (int j = 0; j < lIns.numAttributes() - 1; ++j) {
                    inst.setValue(j, lIns.get(i).value(j));
                }
                for (int j = lIns.numAttributes() - 1; j < aIns.numAttributes() * 2 - 2; ++j) {
                    inst.setValue(j, aIns.get(i).value(k));
                    k++;
                }
                k = 0;
                for (int j = lIns.numAttributes() * 2 - 2; j < bIns.numAttributes() * 3 - 3; ++j) {
                    inst.setValue(j, bIns.get(i).value(k));
                    k++;
                }
                inst.setValue(numAttributes, 0);//Set class as 0
                unlabeled.add(inst);
            }
            unlabeled.setClassIndex(numAttributes);
            return unlabeled;
        }
    }

    /**
     * Creates Instances object based on the features of each color channel after converting the image to Lab and sets class based on provided Ground Truth image
     * @param inputImage RGB input image
     * @param labelImage Label image
     * @param gtImage Ground Truth image
     * @param selectedFeatures ArrayList with selected features from RegionFeatures.Feature
     * @param classes ArrayList of Strings with names of the classes
     * @return Dataset with the features of each color for each region from the labelImage
     */
    public static Instances calculateLabeledColorFeatures(ImagePlus inputImage,
                                                          ImagePlus labelImage,
                                                          ImagePlus gtImage,
                                                          ArrayList<RegionFeatures.Feature> selectedFeatures,
                                                          ArrayList<String> classes)
    {
        ColorSpaceConverter converter = new ColorSpaceConverter();
        ImagePlus lab = converter.RGBToLab(inputImage);
        ImagePlus[] channels = ChannelSplitter.split(lab);
        long startTime = System.currentTimeMillis();
        IJ.log("Calculating channel l features");
        Instances lIns = RegionFeatures.calculateLabeledRegionFeatures(channels[0],labelImage,gtImage,selectedFeatures,classes);
        long elapsedTime = System.currentTimeMillis();
        long estimatedTime = System.currentTimeMillis() - startTime;
        IJ.log( "    Calculating channel l features took " + estimatedTime + " ms");
        startTime = System.currentTimeMillis();
        IJ.log("Calculating channel a features");
        Instances aIns = RegionFeatures.calculateUnlabeledRegionFeatures(channels[1],labelImage,selectedFeatures,classes);
        elapsedTime = System.currentTimeMillis();
        estimatedTime = System.currentTimeMillis() - startTime;
        IJ.log( "    Calculating channel a features took " + estimatedTime + " ms");
        startTime = System.currentTimeMillis();
        IJ.log("Calculating channel b features");
        Instances bIns = RegionFeatures.calculateUnlabeledRegionFeatures(channels[2],labelImage,selectedFeatures,classes);
        elapsedTime = System.currentTimeMillis();
        estimatedTime = System.currentTimeMillis() - startTime;
        IJ.log( "    Calculating channel b features took " + estimatedTime + " ms");
        startTime = System.currentTimeMillis();
        if(lIns==null||aIns==null||bIns==null){
            return null;
        }else {
            for (int i = 0; i < lIns.numAttributes(); ++i) {//all channels should have the same number of attributes
                lIns.renameAttribute(i, lIns.attribute(i).name() + "-L");
                aIns.renameAttribute(i, aIns.attribute(i).name() + "-a");
                bIns.renameAttribute(i, bIns.attribute(i).name() + "-b");
            }
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            int numAttributes = lIns.numAttributes() * 3 - 3;//-3 to remove the class attributes
            for (int i = 0; i < lIns.numAttributes() - 1; ++i) {
                attributes.add(lIns.attribute(i));
            }
            for (int i = 0; i < aIns.numAttributes() - 1; ++i) {
                attributes.add(aIns.attribute(i));
            }
            for (int i = 0; i < bIns.numAttributes() - 1; ++i) {
                attributes.add(bIns.attribute(i));
            }
            attributes.add(new Attribute("Class", classes));
            Instances labeled = new Instances("training data", attributes, 0);
            for (int i = 0; i < lIns.numInstances(); ++i) {
                int k = 0;
                Instance inst = new DenseInstance(numAttributes + 1);
                for (int j = 0; j < lIns.numAttributes() - 1; ++j) {
                    inst.setValue(j, lIns.get(i).value(j));
                }
                for (int j = lIns.numAttributes() - 1; j < aIns.numAttributes() * 2 - 2; ++j) {
                    inst.setValue(j, aIns.get(i).value(k));
                    k++;
                }
                k = 0;
                for (int j = lIns.numAttributes() * 2 - 2; j < bIns.numAttributes() * 3 - 3; ++j) {
                    inst.setValue(j, bIns.get(i).value(k));
                    k++;
                }
                double classValue = lIns.get(i).value(lIns.classIndex());
                inst.setValue(numAttributes,classValue);//Currently taking class of l channel
                labeled.add(inst);
            }
            labeled.setClassIndex(numAttributes);
            elapsedTime = System.currentTimeMillis();
            estimatedTime = System.currentTimeMillis() - startTime;
            IJ.log( "\tCreating instances took" + estimatedTime + "ms");
            return labeled;
        }
    }


}
