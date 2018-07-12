package eus.ehu.tss;


import ij.ImagePlus;
import ij.process.ImageProcessor;
import inra.ijpb.label.LabelImages;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.HashMap;

public class Utils {

    public static Instances merge(Instances data1, Instances data2) throws Exception {
        int asize = data1.numAttributes();
        boolean[] strings_pos = new boolean[asize];

        for(int i = 0; i < asize; ++i) {
            Attribute att = data1.attribute(i);
            strings_pos[i] = att.type() == 2 || att.type() == 1;
        }

        Instances dest = new Instances(data1);
        dest.setRelationName(data1.relationName() + "+" + data2.relationName());
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;

        while(source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            dest.add(instance);

            for(int i = 0; i < asize; ++i) {
                if(strings_pos[i]) {
                    dest.instance(dest.numInstances() - 1).setValue(i, instance.stringValue(i));
                }
            }
        }

        return dest;
    }

    /**
     * Calculates coordinates corresponding to labels in label image
     * @param labelImage input image with labels
     * @return a HashMap where the key is the label and the values are the coordinates of the label
     */
    public static HashMap<Integer,int[]> calculateLabelCoordinates(ImagePlus labelImage){
        HashMap<Integer, Integer> labelIndices = null;
        HashMap<Integer, int[]> result = new HashMap<>();
        final int width = labelImage.getWidth();
        final int height = labelImage.getHeight();

        int[] labels = LabelImages.findAllLabels(labelImage.getImageStack());
        int numLabels = labels.length;
        labelIndices = LabelImages.mapLabelIndices(labels);
        final int numSlices = labelImage.getImageStackSize();
        for( int z=1; z <= numSlices; z++ )
        {
            final ImageProcessor labelsIP = labelImage.getImageStack().getProcessor( z );

            for( int x = 0; x<width; x++ )
                for( int y = 0; y<height; y++ )
                {
                    int labelValue = (int) labelsIP.getPixelValue( x, y );
                    int[] coord = new int[3];
                    coord[0] = x; coord[1] = y; coord[2] = z;
                    result.putIfAbsent(labelValue,coord);
                }
        }
        return result;
    }

}
