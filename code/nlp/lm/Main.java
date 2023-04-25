package nlp.lm;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        String filename = "C:/Users/rhu682/OneDrive/HMC/CS159/final/";
        /* ArrayList<String> sentence = new ArrayList<>();
        sentence.addAll(Arrays.asList("a","b","a","a","c","a","d"));
        LambdaLMModel test = new LambdaLMModel(filename + "data/test1", filename + "data/test_vocab", 0.01);
        System.out.println(test.getPerplexity(filename + "data/test2", 1));
        System.out.println(test.getPerplexity(filename + "data/test2", 2));
        System.out.println(test.getPerplexity(filename + "data/test2", 3)); */

        LambdaLMModel test = new LambdaLMModel(filename + "data/sentences_training", filename + "data/sentences_vocab", 0.01);
        System.out.println(test.getPerplexity(filename + "data/sentences_test", 1));
        System.out.println(test.getPerplexity(filename + "data/sentences_test", 2));
        System.out.println(test.getPerplexity(filename + "data/sentences_test", 3));
      }
}
