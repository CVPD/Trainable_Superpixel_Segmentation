package eus.ehu.tss;


import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.ResultsTable;
import ij.process.ImageConverter;
import ij.process.ImageProcessor;
import ij.process.StackConverter;
import inra.ijpb.data.image.Images3D;
import inra.ijpb.label.LabelImages;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.HashMap;

public class Utils {

    /**
     * Merge two instances
     * @param data1
     * @param data2
     * @return Instances object containing instances of both datasets
     * @throws Exception instances mismatch
     */
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

    /**
     * Merge two Results Tables assuming both have the same columns
     * @param rs1
     * @param rs2
     * @return resulting resultstable
     */
    public static ResultsTable mergeResultsTables(ResultsTable rs1, ResultsTable rs2){
        ResultsTable result = (ResultsTable) rs1.clone();
        for(int i=0;i<rs2.getCounter();++i){
            result.incrementCounter();
            result.addLabel(rs2.getLabel(i));
            for(int j=0;j<rs2.getLastColumn();++j){
                result.addValue(j,rs2.getValueAsDouble(j,i));
            }
        }
        return result;
    }

    /**
     * Remaps label image so that regions in different slices have different values
     * @param labelImage label image
     * @return label image with different regions each slice
     */
    public static ImagePlus remapLabelImage(ImagePlus labelImage){
        ImageStack img = labelImage.getStack();
        double max = 0;
        double prevMax = 0;
        for(int z=0;z<img.getSize();++z){
            ImageProcessor slice = img.getProcessor(z+1);
            LabelImages.remapLabels(slice);
            for(int y = 0; y < slice.getHeight();++y){
                for(int x = 0;x<slice.getWidth();++x){
                    double p = slice.getf(x,y);
                    if(p!=0) {
                        if (p > max) {
                            max = p;
                        }
                        img.setVoxel(x, y, z, p + prevMax);
                    }
                }
            }
            prevMax+=max;
            max=0;
        }
        ImagePlus result = new ImagePlus(labelImage.getShortTitle(),img);
        Images3D.optimizeDisplayRange(result);
        return result;
    }

    /**
     * Convert image to 8 bit in place without scaling it. (Taken from Weka_Segmentation.)
     *
     * @param image input image
     */
    public static void convertTo8bitNoScaling( ImagePlus image )
    {
        boolean aux = ImageConverter.getDoScaling();

        ImageConverter.setDoScaling( false );

        if( image.getImageStackSize() > 1)
            (new StackConverter( image )).convertToGray8();
        else
            (new ImageConverter( image )).convertToGray8();

        ImageConverter.setDoScaling( aux );
    }

}
