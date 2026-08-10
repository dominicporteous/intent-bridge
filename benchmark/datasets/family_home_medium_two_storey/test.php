<?php

use Symfony\Component\Yaml\Yaml;
use Symfony\Component\Yaml\Exception\ParseException;

require_once("/home/boxer/devel/syn/levis/vendor/autoload.php");

$replace = [
    'bedroom_1' => 'bedroom 1',
    'bedroom1' => 'bedroom 1',
    'bedroom_2' => 'bedroom 2',
    'bedroom2' => 'bedroom 2',
    'bedroom_3' => 'bedroom 3',
    'bedroom3' => 'bedroom 3',
    'bedroom_4' => 'bedroom 4',
    'bedroom4' => 'bedroom 4',
    'family_room' => 'family room',
    'main_bath' => 'main bathroom',
    'master_bath' => 'master bathroom',
    'master_bedroom' => 'master bedroom',
    'powder_room' => 'powder room',
    'script.good_night' => 'good night script',
    'scene.good_morning' => 'good morning scene',
    'scene.movie_night' => 'movie night scene',
    'script.leaving home' => 'leaving home script',
    'main bath ' => 'main bathroom ',
    'master bath ' => 'master bathroom ',
    'scene.dinner_time' => 'dinner time scene',
    'scene.kids_bedtime' => "kids bedtime scene"
];

$dir = "devices";
$files = scandir("./$dir");
foreach ($files as $file) {
    if (!str_ends_with($file, '.yaml')) { continue; }
    $data = yaml_parse_file("./$dir/$file");

        $index=0;
    foreach ($data as  $vars) {

        $s = [];
        foreach ($vars['sentences'] as $text) {
            $text = strtr($text, $replace);
            if ($file == 'scenes.yaml' && !str_contains($text, 'scene')) {
                $text = str_ireplace('movie night', 'movie night scene', $text);
                $text = str_ireplace('kids bedtime', 'kids bedtime scene', $text);
                $text = str_ireplace('dinner time', 'dinner time scene', $text);
                $text = str_ireplace('good morning', 'good morning scene', $text);
            } else if ($file == 'scripts.yaml' && !str_contains($text, 'script')) {
                $text = str_ireplace('good night', 'good night script', $text);
                $text = str_ireplace('leaving home', 'leaving home script', $text);
            }
            $s[] = $text;
        }
        $data[$index]['sentences'] = $s;
        $index++;
    }

    $yaml = Yaml::dump($data, 8, 2);
    file_put_contents("./$dir/$file", $yaml);
    echo "$file\n";

}

