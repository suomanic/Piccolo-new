#version 310 es

#extension GL_GOOGLE_include_directive : enable

#include "constants.h"
//layout(input_attachment_index = 0, set = 0, binding = 0) uniform highp subpassInput in_color;
layout(set = 0, binding = 0) uniform sampler2D in_color;

layout(location = 0) in highp vec2 in_uv;

layout(location = 0) out highp vec4 out_color;

//layout(set = 0, binding = 2) uniform sampler2D texSampler;

highp float rand(highp vec2 uv){
 return fract(sin(dot(uv, vec2(12.9898,78.233)))*43578.5453);   
}
highp float randomNoise(highp float x, highp float y)
{
    return fract(sin(dot(vec2(x, y), vec2(12.9898, 78.233))) * 43758.5453);
}
#define NOISE_SIMPLEX_1_DIV_289 0.00346020761245674740484429065744f
//#define _Frequency 2.3
#define _RGBSplit 37.8
#define _Speed 0.32
//#define _Amount 1.0

#define _Amount 0.13
#define _Frequency 1.5
#define _JitterIntensity 0.45
highp vec3 taylorInvSqrt(highp vec3 r){return 1.79284291400159 - 0.85373472095314 * r;}

highp vec2 mod289(highp vec2 x)
{
	return x - floor(x s* NOISE_SIMPLEX_1_DIV_289) * 289.0;
}

highp vec3 mod289(highp vec3 x)
{
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}

highp vec3 permute(highp vec3 x)
{
	return mod289(x * x * 34.0 + x);
}


highp float snoise(highp vec2 v)
{
	const highp vec4 C = vec4(0.211324865405187, // (3.0-sqrt(3.0))/6.0
	0.366025403784439, // 0.5*(sqrt(3.0)-1.0)
	- 0.577350269189626, // -1.0 + 2.0 * C.x
	0.024390243902439); // 1.0 / 41.0
	// First corner
	highp vec2 i = floor(v + dot(v, C.yy));
	highp vec2 x0 = v - i + dot(i, C.xx);
	
	// Other corners
	highp vec2 i1;
	i1.x = step(x0.y, x0.x);
	i1.y = 1.0 - i1.x;
	
	// x1 = x0 - i1  + 1.0 * C.xx;
	// x2 = x0 - 1.0 + 2.0 * C.xx;
	highp vec2 x1 = x0 + C.xx - i1;
	highp vec2 x2 = x0 + C.zz;
	
	// Permutations
	i = mod289(i); // Avoid truncation effects in permutation
	highp vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
	+ i.x + vec3(0.0, i1.x, 1.0));
	
	highp vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x1, x1), dot(x2, x2)), 0.0);
	m = m * m;
	m = m * m;
	
	// Gradients: 41 points uniformly over a line, mapped onto a diamond.
	// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)
	highp vec3 x = 2.0 * fract(p * C.www) - 1.0;
	highp vec3 h = abs(x) - 0.5;
	highp vec3 ox = floor(x + 0.5);
	highp vec3 a0 = x - ox;
	
	// Normalise gradients implicitly by scaling m
	m *= taylorInvSqrt(a0 * a0 + h * h);
	
	// Compute final noise value at P
	highp vec3 g;
	g.x = a0.x * x0.x + h.x * x0.y;
	g.y = a0.y * x1.x + h.y * x1.y;
	g.z = a0.z * x2.x + h.z * x2.y;
	return 130.0 * dot(m, g);
}
void main()
{
    //highp vec4 color       = subpassLoad(in_color).rgba;

    int x = 0;
   /*
    0：边角变暗
    1：均值模糊
    2：高斯模糊
    3：径向模糊
    4: 移轴模糊
    5: 光圈模糊
    6：粒状模糊
    7：双重模糊
    8: RGB分离
    9: 波动
    10：扫描线抖动
    */
    if(x == -1)
    {
        highp vec4 color = texture(in_color, (in_uv+0.25f) / 1.5f);
        out_color = color;
    }
    if(x == 0)
    {
        //highp vec4 color = texture(in_color, (in_uv+0.25f) / 1.5f);
        highp vec4 color = texture(in_color, in_uv);
        highp float cutoff = 0.4f;
        highp float exponent =  2.5f;
        highp float len = length(in_uv - 0.5f);
        highp float ratio = pow(cutoff / len, exponent);
        out_color = color * min(ratio, 1.0f);
    }
    else if(x == 1)
    {
        highp vec4 sample0, sample1, sample2, sample3;
        highp float fStep = 0.002;
        sample0 = texture(in_color, vec2(in_uv.x - fStep, in_uv.y - fStep));
        sample1 = texture(in_color, vec2(in_uv.x + fStep, in_uv.y + fStep));
        sample2 = texture(in_color, vec2(in_uv.x + fStep, in_uv.y - fStep));
        sample3 = texture(in_color, vec2(in_uv.x - fStep, in_uv.y + fStep));
        out_color = (sample0 + sample1 + sample2 + sample3) / 4.0f;
        //out_color = texture(in_color, (in_uv+0.25f) / 1.5f);
    }
    else if(x == 2)
    {
        /*Kernel[6] = 1.0; Kernel[7] = 2.0; Kernel[8] = 1.0;
        Kernel[3] = 2.0; Kernel[4] = 4.0; Kernel[5] = 2.0;
        Kernel[0] = 1.0; Kernel[1] = 2.0; Kernel[2] = 1.0;*/
        /*Offset[0] = vec2(-fStep,-fStep); Offset[1] = vec2(0.0f,-fStep); Offset[2] = vec2(fStep,-fStep);
        Offset[3] = vec2(-fStep,0.0f);    Offset[4] = vec2(0.0f, 0.0f);   Offset[5] = vec2(fStep, 0.0f);
        Offset[6] = vec2(-fStep, fStep); Offset[7] = vec2(0.0f, fStep); Offset[8] = vec2(fStep, fStep);*/
        
        const int KernelSize = 25;
        highp float stepValue = 0.0015;
        highp vec4 sum = vec4(0.0f);
        highp float Kernel[KernelSize];
        
        Kernel[20] = 1.0; Kernel[21] = 4.0;  Kernel[22] = 6.0,  Kernel[23] = 4.0,  Kernel[24] = 1.0;
        Kernel[15] = 4.0; Kernel[16] = 16.0; Kernel[17] = 24.0, Kernel[18] = 16.0, Kernel[19] = 4.0;
        Kernel[10] = 6.0; Kernel[11] = 24.0; Kernel[12] = 36.0, Kernel[13] = 24.0, Kernel[14] = 6.0;
        Kernel[5] = 4.0;  Kernel[6]  = 16.0; Kernel[7]  = 24.0, Kernel[8]  = 16.0, Kernel[9]  = 4.0;
        Kernel[0] = 1.0;  Kernel[1]  = 4.0;  Kernel[2]  = 6.0,  Kernel[3]  = 4.0,  Kernel[4]  = 1.0;
        highp float fStep = stepValue;
        highp vec2 Offset[KernelSize];
        Offset[20] = vec2(-2.0f*fStep,2.0f*fStep);  Offset[21] = vec2(-fStep,2.0f*fStep); Offset[22] = vec2(0.0f, 2.0f*fStep); Offset[23] = vec2(fStep,2.0f*fStep); Offset[24] = vec2(2.0f*fStep,2.0f*fStep);
        Offset[15] = vec2(-2.0f*fStep,fStep);       Offset[16] = vec2(-fStep,fStep);      Offset[17] = vec2(0.0f, fStep);      Offset[18] = vec2(fStep,fStep);      Offset[19] = vec2(2.0f*fStep,fStep);
        Offset[10] = vec2(-2.0f*fStep,0.0f);        Offset[11] = vec2(-fStep, 0.0f);      Offset[12] = vec2(0.0f, 0.0f);       Offset[13] = vec2(fStep,0.0f);       Offset[14] = vec2(2.0f*fStep,0.0f);
        Offset[5]  = vec2(-2.0f*fStep, -fStep);     Offset[6] = vec2(-fStep, -fStep);     Offset[7]  = vec2(0.0f, -fStep);     Offset[8] = vec2(fStep,-fStep);      Offset[9] = vec2(2.0f*fStep,-fStep);
        Offset[0]  = vec2(-2.0f*fStep, -2.0f*fStep);Offset[1] = vec2(-fStep,-2.0f*fStep); Offset[2]  = vec2(0.0f, -2.0f*fStep);Offset[3] = vec2(fStep,-2.0f*fStep); Offset[4] = vec2(2.0f*fStep,-2.0f*fStep);
        int i;
        for (i = 0; i < KernelSize; i++)
        {
            highp vec4 tmp = texture(in_color, in_uv + Offset[i]);
            sum += tmp * Kernel[i];
        }
        out_color = sum / 256.0f;
    }
    else if(x == 3)
    {
        const int nsamples = 30;
        highp vec2 center =  in_uv / 2.0f + vec2(0.25f, 0.25f); 
        highp float blurStart = 1.0;
        highp float blurWidth = 0.1;

        
        highp vec2 uv = in_uv;
        
        uv -= center;
        highp float precompute = blurWidth * (1.0 / float(nsamples - 1));
        
        highp vec4 color = vec4(0.0);
        for(int i = 0; i < nsamples; i++)
        {
            highp float scale = blurStart + (float(i)* precompute);
            color += texture(in_color, uv * scale + center);
        }
        
        
        color /= float(nsamples);
        
        out_color = color;
    }
    else if(x == 4)
    {
        const highp float bluramount = 0.7;
        const highp float center = 1.0;
        const highp float stepSize = 0.004;
        const highp float steps = 15.0;
        const highp float minOffs = (float(steps - 1.0)) / -2.0;
        const highp float maxOffs = (float(steps - 1.0)) / +2.0;
        highp vec3 c = texture(in_color, in_uv).rgb;
        highp vec2 tcoord = in_uv;
        highp float amount;
        highp vec4 blurred;
        amount = pow((tcoord.y * center) * 2.0 - 1.0, 2.0) * bluramount;
        blurred = vec4(0.0, 0.0, 0.0, 1.0);
        for (highp float offsX = minOffs; offsX <= maxOffs; ++offsX)
        {
            for (highp float offsY = minOffs; offsY <= maxOffs; ++offsY)
            {
                highp vec2 temp_tcoord = tcoord.xy;
                temp_tcoord.x += offsX * amount * stepSize;
                temp_tcoord.y += offsY * amount * stepSize;
                blurred += texture(in_color, temp_tcoord);
            } 
        } 
        blurred /= float(steps * steps);
        out_color = blurred;
    }
    else if(x == 5)
    {

        const int KernelSize = 25;
        highp float stepValue = 0.005;
        highp vec4 sum = vec4(0.0f);
        highp float Kernel[KernelSize];
        
        Kernel[20] = 1.0; Kernel[21] = 4.0;  Kernel[22] = 6.0,  Kernel[23] = 4.0,  Kernel[24] = 1.0;
        Kernel[15] = 4.0; Kernel[16] = 16.0; Kernel[17] = 24.0, Kernel[18] = 16.0, Kernel[19] = 4.0;
        Kernel[10] = 6.0; Kernel[11] = 24.0; Kernel[12] = 36.0, Kernel[13] = 24.0, Kernel[14] = 6.0;
        Kernel[5] = 4.0;  Kernel[6]  = 16.0; Kernel[7]  = 24.0, Kernel[8]  = 16.0, Kernel[9]  = 4.0;
        Kernel[0] = 1.0;  Kernel[1]  = 4.0;  Kernel[2]  = 6.0,  Kernel[3]  = 4.0,  Kernel[4]  = 1.0;
        highp float fStep = stepValue;
        highp vec2 Offset[KernelSize];
        Offset[20] = vec2(-2.0f*fStep,2.0f*fStep);  Offset[21] = vec2(-fStep,2.0f*fStep); Offset[22] = vec2(0.0f, 2.0f*fStep); Offset[23] = vec2(fStep,2.0f*fStep); Offset[24] = vec2(2.0f*fStep,2.0f*fStep);
        Offset[15] = vec2(-2.0f*fStep,fStep);       Offset[16] = vec2(-fStep,fStep);      Offset[17] = vec2(0.0f, fStep);      Offset[18] = vec2(fStep,fStep);      Offset[19] = vec2(2.0f*fStep,fStep);
        Offset[10] = vec2(-2.0f*fStep,0.0f);        Offset[11] = vec2(-fStep, 0.0f);      Offset[12] = vec2(0.0f, 0.0f);       Offset[13] = vec2(fStep,0.0f);       Offset[14] = vec2(2.0f*fStep,0.0f);
        Offset[5]  = vec2(-2.0f*fStep, -fStep);     Offset[6] = vec2(-fStep, -fStep);     Offset[7]  = vec2(0.0f, -fStep);     Offset[8] = vec2(fStep,-fStep);      Offset[9] = vec2(2.0f*fStep,-fStep);
        Offset[0]  = vec2(-2.0f*fStep, -2.0f*fStep);Offset[1] = vec2(-fStep,-2.0f*fStep); Offset[2]  = vec2(0.0f, -2.0f*fStep);Offset[3] = vec2(fStep,-2.0f*fStep); Offset[4] = vec2(2.0f*fStep,-2.0f*fStep);
        int i;
        for (i = 0; i < KernelSize; i++)
        {
            highp vec4 tmp = texture(in_color, in_uv + Offset[i]);
            sum += tmp * Kernel[i];
        }
        highp vec4 blur_color = sum / 256.0f;




        highp float height = float(textureSize(in_color, 0).y);
        highp float width = float(textureSize(in_color, 0).y);
        highp vec3 line1 = vec3(1.0f, 0.0f, -0.5f);
        highp vec3 line2 = vec3(1.0f, 0.0f, 0.5f);
        
        highp float inner = 0.1f;
        highp float outer = 1.0f;
        highp float intensity = 0.5f;


        highp vec2 center =  vec2(0.5f, 0.5f);
        highp vec4 originalColor = texture(in_color, in_uv);
        highp vec4 tempColor;
        highp float ratio = height / width;
        highp vec2 ellipse = vec2(1, ratio * ratio);
        highp float fx = (in_uv.x - center.x);
        highp float fy = (in_uv.y - center.y);
        highp float dist = sqrt(fx * fx * ellipse.x + fy * fy * ellipse.y);
        if (dist < inner || dist > outer) {
            tempColor = originalColor;
        } 
        else 
        {
            highp vec4 blurColor = blur_color;
            highp float alpha = (dist - inner) * 10.0f;
            alpha = clamp(alpha, 0.0, 1.0);
            tempColor = mix(originalColor, blurColor, alpha);
        }
        out_color = mix(originalColor, tempColor, intensity);
    }
    else if(x == 6)
    {
        highp vec2 uv = in_uv;
        highp vec4 t = vec4(0.0);
        highp float dist = 4.0;
        highp vec2 texel = vec2(1.0 / 3000.0, 1.0 / 3000.0);
        highp vec2 d = texel * dist;
        int loops = 10;
        for(int i = 0; i < loops; i++){
            
            highp float r1 = clamp(rand(uv * float(i))*2.0-1.0, -d.x, d.x);
            highp float r2 = clamp(rand(uv * float(i+loops))*2.0-1.0, -d.y, d.y);
        
            t += texture(in_color, uv + vec2(r1, r2));
        }
        
        t /= float(loops);

        out_color = t;
    }
    else if(x == 7)
    {
        highp vec2 uv = in_uv;
        highp vec2 halfpixel = 0.5 / (vec2(float(textureSize(in_color, 0).x), 
        float(textureSize(in_color, 0).y)) / 2.0);
        highp float offset = 3.0;

        highp vec4 sum = texture(in_color, uv) * 4.0;
        sum += texture(in_color, uv - halfpixel.xy * offset);
        sum += texture(in_color, uv + halfpixel.xy * offset);
        sum += texture(in_color, uv + vec2(halfpixel.x, -halfpixel.y) * offset);
        sum += texture(in_color, uv - vec2(halfpixel.x, -halfpixel.y) * offset);

        sum += texture(in_color, uv + vec2(-halfpixel.x * 2.0, 0.0) * offset);
        sum += texture(in_color, uv + vec2(-halfpixel.x, halfpixel.y) * offset) * 2.0;
        sum += texture(in_color, uv + vec2(0.0, halfpixel.y * 2.0) * offset);
        sum += texture(in_color, uv + vec2(halfpixel.x, halfpixel.y) * offset) * 2.0;
        sum += texture(in_color, uv + vec2(halfpixel.x * 2.0, 0.0) * offset);
        sum += texture(in_color, uv + vec2(halfpixel.x, -halfpixel.y) * offset) * 2.0;
        sum += texture(in_color, uv + vec2(0.0, -halfpixel.y * 2.0) * offset);
        sum += texture(in_color, uv + vec2(-halfpixel.x, -halfpixel.y) * offset) * 2.0;

        out_color = sum / 20.0; //12.0;
    }
    else if(x == 8)
    {
        highp vec2 uv = in_uv;
    
        highp float splitAmount = 0.01 * randomNoise(2.0, 2.0);

        highp vec4 ColorR = texture(in_color, vec2(uv.x + splitAmount, uv.y));
        highp vec4 ColorG = texture(in_color, uv);
        highp vec4 ColorB = texture(in_color, vec2(uv.x - splitAmount, uv.y));

        out_color = vec4(ColorR.r, ColorG.g, ColorB.b,1.0);
    }
    else if(x == 9)
    {
        highp float strength = 1.0;
        highp vec2 uv = in_uv;
        highp float noise_wave_1 = snoise(vec2(uv.y, 2.0*_Speed)) * (strength * _Amount * 32.0);
        highp float noise_wave_2 = snoise(vec2(uv.y, 1.0*_Speed)) * (strength * _Amount * 4.0);
        highp float noise_wave_x = noise_wave_1 * noise_wave_2 / float(textureSize(in_color, 0).x);
        highp float uv_x = uv.x + noise_wave_x;
        
        highp float rgbSplit_uv_x = (_RGBSplit * 50.0 + (20.0 * strength + 1.0))
        * noise_wave_x / float(textureSize(in_color, 0).x);
        
        highp vec4 colorG = texture(in_color, vec2(uv_x, uv.y));
        highp vec4 colorRB = texture(in_color, vec2(uv_x + rgbSplit_uv_x, uv.y));

        out_color = vec4(colorRB.r, colorG.g, colorRB.b, colorRB.a + colorG.a);
    }
    else if(x == 10)
    {
        highp vec2 uv = in_uv;

        highp float strength = 0.5 + 0.5 * cos(0.5 * _Frequency);	
        highp float jitter = randomNoise(uv.y, 0.5) * 2.0 - 1.0;
        highp float threshold = clamp(1.0f - _JitterIntensity * 1.2f, 0.0, 1.0);
        jitter *= step(threshold, abs(jitter)) * _Amount * strength;	

        // Output to screen
        out_color = texture(in_color, fract(uv + vec2(jitter, 0)));	
    }
}


    
