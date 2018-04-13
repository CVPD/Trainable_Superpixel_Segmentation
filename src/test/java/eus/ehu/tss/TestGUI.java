package eus.ehu.tss;


import ij.IJ;
import ij.ImageJ;

public class TestGUI {
    public static void main( final String[] args )
    {
        ImageJ.main( args );
        new Trainable_Superpixel_Segmentation().run(null);
    }
}
