import { defineCollection, z } from 'astro:content';
import { file, glob } from "astro/loaders";

// const blogs = defineCollection({
//   loader: file('src/data/blog-data.json'),
//   schema: z.object({
//     title: z.string(),
//     description: z.string(),
//     slug: z.string(),
//     published_date: z.string()

//   })
// });


const blogs = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "src/data/blogs" }),
  schema: z.object({
    title: z.string(), 
    description: z.string(), 
    published_date: z.string(),
    toc: z.array(z.string()),
    githubURL: z.string().optional()
  })
});


export const collections =  { blogs }